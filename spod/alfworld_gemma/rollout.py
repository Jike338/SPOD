"""vLLM student policy for one rollout/evaluation phase (fresh process per phase, weights loaded from disk)."""
import os,sys,json,time,subprocess,signal
from collections import defaultdict,deque
from pathlib import Path
from common import *
from protocol import SYSTEM
from pipeline_io import generation_prompt, extract_action
from fixed_protocol import request_seed


ROLE=os.environ.get('ALF_ROLE','rollout0')
cache=Path(os.environ['TMPDIR']).parent/'kernel_cache'/'triton'/ROLE;cache.mkdir(parents=True,exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(cache)

def main():
 from transformers import AutoTokenizer, GenerationConfig
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 status(ROLE,stage='loading')
 model_path=CFG.get('eval_model',CFG['student'])
 tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 stop_ids=GenerationConfig.from_pretrained(model_path,local_files_only=True).eos_token_id
 if isinstance(stop_ids,int):stop_ids=[stop_ids]
 llm=LLM(model=model_path,dtype='bfloat16',tensor_parallel_size=1,max_model_len=16384,max_num_seqs=72,max_num_batched_tokens=8192,gpu_memory_utilization=CFG.get('rollout_memory',.30),enforce_eager=True,enable_prefix_caching=True,limit_mm_per_prompt={'image':0,'video':0,'audio':0},seed=CFG['seed'],structured_outputs_config=dict(backend='xgrammar'))
 version=int(CFG.get("policy_version",0))
 bridge=None;bridge_log=None
 save(ROOT/(ROLE+'_ready.json'),dict(time=time.time(),pid=os.getpid()))
 def episodes(req):
  assert req['version']==version,(req['version'],version)
  max_steps=int(req.get('max_steps',CFG['max_steps']));assert 1<=max_steps<=CFG['max_steps']
  rows=req['rows'];out=ROOT/req['out'];out.mkdir(parents=True,exist_ok=True)
  if (out/'completion.json').exists():
   complete=json.loads((out/'completion.json').read_text())
   assert complete['version']==version and complete['tasks']==len(rows)
   for q in rows:
    prior=json.loads((out/(q['id']+'.json')).read_text());assert prior['policy_version']==version and prior['game']==q['game'] and prior['seed']==req['seed']
   return dict(**complete,reused_saved_rollouts=True)
  nonlocal bridge,bridge_log
  if bridge is None:
   bridge_log=(ROOT/(ROLE+'_environment.log')).open('a')
   bridge=subprocess.Popen([os.environ.get('ALFWORLD_PYTHON',sys.executable),'-u',str(CODE/'alfworld_bridge.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=bridge_log,text=True,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  p=bridge
  def ask(q):
   tick=time.time();p.stdin.write(json.dumps(q)+'\n');p.stdin.flush();line=p.stdout.readline();assert line,'Environment bridge exited';v=json.loads(line);assert 'error' not in v,v;timings['environment_'+q['op']]=timings.get('environment_'+q['op'],0.)+time.time()-tick;return v
  results=[];started=time.time();timings=dict(environment=0.,generation=0.,prompt=0.,write=0.);actions_generated=0;input_tokens=0;output_tokens=0;chunk=int(req.get('concurrency',18))
  try:
   for lo in range(0,len(rows),chunk):
    batch=rows[lo:lo+chunk]
    tick=time.time();state=ask(dict(op='reset',games=[q['game'] for q in batch],seed=req['seed'],max_steps=max_steps))
    timings['environment']+=time.time()-tick
    bygame=defaultdict(deque)
    for q in batch:bygame[q['game']].append(q)
    order=[bygame[x].popleft() for x in state['games']];initial=list(state['obs']);hist=[[] for _ in order];done=[False]*len(order);won=[False]*len(order)
    for turn in range(max_steps):
     check();active=[i for i in range(len(order)) if not done[i] and len(hist[i])<max_steps]
     if not active:break
     tick=time.time();prompts=[];params=[];message_batches=[]
     for i in active:
      previous=hist[i]
      # Keep initial/current observations and command list; remove oldest history only if necessary.
      while True:
       h='\n'.join('Action: '+x['action']+'\nObservation: '+x['observation'] for x in previous if x['action'] is not None)
       content='Initial observation:\n'+initial[i]+'\nHistory:\n'+h+'\nCurrent observation:\n'+state['obs'][i]+'\nAvailable commands:\n'+'\n'.join(state['commands'][i])
       messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)]
       ids=generation_prompt(tok,messages)
       if len(ids)+CFG['response_tokens']<=CFG['context_limit']:break
       assert previous,'Initial/current observation cannot fit';previous=previous[1:]
      prompts.append(dict(prompt_token_ids=ids));message_batches.append(messages)
      params.append(SamplingParams(temperature=req['temperature'],top_p=.95 if req['temperature'] else 1.0,top_k=20 if req['temperature'] else -1,max_tokens=CFG['response_tokens'],seed=request_seed(req['seed'],order[i]['id'],len(hist[i])),skip_special_tokens=False,structured_outputs=StructuredOutputsParams(choice=state['commands'][i])))
     timings['prompt']+=time.time()-tick;tick=time.time()
     generated=llm.generate(prompts,params,use_tqdm=False)
     timings['generation']+=time.time()-tick
     actions=['look']*len(order);raws={};accepted=[]
     for i,x,pr,messages in zip(active,generated,prompts,message_batches):
      o=x.outputs[0];tt=list(o.token_ids);assert tt
      action=extract_action(tok,tt,state['commands'][i],stop_ids)
      raws[i]=dict(request_seed=request_seed(req['seed'],order[i]['id'],len(hist[i])),base_seed=req['seed'],text=o.text,messages=messages,prompt_token_ids=pr['prompt_token_ids'],output_token_ids=tt,loss_mask=[1]*len(tt),policy_version=version,finish=o.finish_reason,truncated=o.finish_reason=='length',extracted_action=action,response_tokens=len(tt))
      input_tokens+=len(pr['prompt_token_ids']);output_tokens+=len(tt)
      if action is not None:
       actions[i]=action;accepted.append(i);actions_generated+=1
      else:done[i]=True
     # Fillers affect only already-terminated vector slots and never enter their history or score.
     tick=time.time();nxt=ask(dict(op='step',actions=actions)) if accepted else state;timings['environment']+=time.time()-tick
     for i in active:
      valid=raws[i]['extracted_action'] is not None
      observation=nxt['obs'][i] if valid else state['obs'][i]
      if valid:won[i]=bool(nxt['won'][i]);done[i]=done[i] or won[i] or nxt['dones'][i] or len(hist[i])+1>=max_steps
      hist[i].append(dict(turn=len(hist[i])+1,action=raws[i]['extracted_action'],raw=raws[i],available_commands=state['commands'][i],observation=observation,won=won[i],unclear=not valid))
      # Preserve failed/unclear responses too: teacher KL does not require an executable action.
      # Histories stay in the existing in-memory batch; persist each final episode once below.
     save(ROOT/'latest_turn.json',dict(version=version,out=req['out'],turn=turn+1,batch=lo,active=len(active),response_tokens=[raws[i]['response_tokens'] for i in active],accepted=len(accepted),unclear=len(active)-len(accepted),truncated=sum(raws[i]['truncated'] for i in active),elapsed=time.time()-started))
     state=nxt;status(ROLE,stage='episodes',version=version,out=req['out'],batch=lo,turn=turn+1,active=sum(not x for x in done),elapsed=time.time()-started)
    for q,hs,success,ob in zip(order,hist,won,initial):
     result=dict(**q,history=hs,success=success,initial=ob,policy_version=version,seed=req['seed'],turns=len(hs));write_tick=time.time();save(out/(q['id']+'.json'),result);timings['write']+=time.time()-write_tick;results.append(result)
   completion=dict(unclear_turns=sum(h['unclear'] for x in results for h in x['history']),truncated_turns=sum(h['raw']['truncated'] for x in results for h in x['history']),responses=sum(len(x['history']) for x in results),timings=timings,concurrency=chunk,actions=actions_generated,input_tokens=input_tokens,output_tokens=output_tokens,tasks=len(results),success=sum(x['success'] for x in results),seconds=time.time()-started,version=version)
   save(out/'completion.json',completion);return completion
  finally:
   pass # Keep the CPU environment service resident across batches.
 def handle(req):
  op=req['op']
  if op=='episodes':return episodes(req)
  elif op=='shutdown':return dict(ok=True)
  else:raise ValueError(op)
 try:serve(ROLE,handle)
 finally:
  llm.llm_engine.engine_core.shutdown()
  if bridge is not None and bridge.poll() is None:
   import select,threading
   bridge.stdin.write(json.dumps(dict(op='close'))+'\n');bridge.stdin.flush()
   if select.select([bridge.stdout],[],[],10)[0]:bridge.stdout.readline()
   bridge.stdin.close()
   try:bridge.wait(timeout=5)
   except subprocess.TimeoutExpired:
    os.killpg(bridge.pid,signal.SIGTERM)
    try:bridge.wait(timeout=5)
    except subprocess.TimeoutExpired:
     os.killpg(bridge.pid,signal.SIGKILL)
     threading.Thread(target=bridge.wait,daemon=True).start()
  if bridge_log is not None:bridge_log.close()
if __name__=='__main__':guarded(main,'rollout')
