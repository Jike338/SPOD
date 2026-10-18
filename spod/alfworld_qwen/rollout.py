"""Persistent vLLM student policy; trainer weights are loaded from a BF16 snapshot between rollout batches."""
import os,sys,json,time,subprocess,signal
from common import *
from prompts import SYSTEM

ROLE=os.environ.get('ALF_ROLE','rollout0')
cache=ROOT/'runtime_cache'/'triton'/ROLE;cache.mkdir(parents=True,exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(cache)

def load_snapshot(worker,path):
 from safetensors import safe_open
 with safe_open(path,framework='pt',device='cpu') as handle:
  worker.model_runner.model.load_weights((name,handle.get_tensor(name)) for name in handle.keys())

def main():
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 status(ROLE,stage='loading')
 tok=AutoTokenizer.from_pretrained(CFG['student'],local_files_only=True)
 llm=LLM(model=CFG['student'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=16384,max_num_seqs=72,max_num_batched_tokens=8192,gpu_memory_utilization=.30,enforce_eager=True,enable_prefix_caching=True,limit_mm_per_prompt={'image':0,'video':0},seed=CFG['seed'],structured_outputs_config=dict(backend='xgrammar'))
 version=0
 bridge=None;bridge_log=None
 save(ROOT/(ROLE+'_ready.json'),dict(time=time.time(),pid=os.getpid()))
 def episodes(req):
  assert req['version']==version,(req['version'],version)
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
    batch=rows[lo:lo+chunk];tick=time.time();state=ask(dict(op='reset',games=[q['game'] for q in batch],seed=req['seed'],max_steps=CFG['max_steps']))
    timings['environment']+=time.time()-tick
    from collections import defaultdict,deque
    bygame=defaultdict(deque)
    for q in batch:bygame[q['game']].append(q)
    order=[bygame[x].popleft() for x in state['games']];initial=list(state['obs']);hist=[[] for _ in order];done=[False]*len(order);won=[False]*len(order)
    for turn in range(CFG['max_steps']):
     check();active=[i for i in range(len(order)) if not done[i] and len(hist[i])<CFG['max_steps']]
     if not active:break
     tick=time.time();prompts=[];params=[]
     for i in active:
      previous=hist[i]
      # Keep initial/current observations and command list; remove oldest history only if necessary.
      while True:
       h='\n'.join('Action: '+x['action']+'\nObservation: '+x['observation'] for x in previous)
       content='Initial observation:\n'+initial[i]+'\nHistory:\n'+h+'\nCurrent observation:\n'+state['obs'][i]+'\nAvailable commands:\n'+'\n'.join(state['commands'][i])
       ids=tok.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
       if len(ids)+80<=CFG['context_limit']:break
       assert previous,'Initial/current observation cannot fit';previous=previous[1:]
      prompts.append(dict(prompt_token_ids=ids))
      params.append(SamplingParams(temperature=.6,top_p=.95,top_k=20,max_tokens=80,seed=req['seed'],skip_special_tokens=False,structured_outputs=StructuredOutputsParams(choice=state['commands'][i])))
     timings['prompt']+=time.time()-tick;tick=time.time()
     gen=llm.generate(prompts,params,use_tqdm=False);timings['generation']+=time.time()-tick;actions=['look']*len(order);raws={}
     for i,x,pr in zip(active,gen,prompts):
      o=x.outputs[0];actions_generated+=1;input_tokens+=len(pr['prompt_token_ids']);output_tokens+=len(o.token_ids);action=o.text.replace('<|im_end|>','').replace('<|endoftext|>','').strip()
      if action.startswith('Action:'):action=action[7:].strip()
      action=action.strip('`').strip();assert action in state['commands'][i],(action,state['commands'][i]);actions[i]=action
      raws[i]=dict(text=o.text,prompt_token_ids=pr['prompt_token_ids'],output_token_ids=list(o.token_ids),finish=o.finish_reason,policy_version=version)
     tick=time.time();nxt=ask(dict(op='step',actions=actions));timings['environment']+=time.time()-tick
     for i in active:
      hist[i].append(dict(turn=len(hist[i])+1,action=actions[i],raw=raws[i],available_commands=state['commands'][i],observation=nxt['obs'][i],won=nxt['won'][i]))
      won[i]=nxt['won'][i];done[i]=won[i] or nxt['dones'][i] or len(hist[i])>=CFG['max_steps']
     state=nxt;status(ROLE,stage='episodes',version=version,out=req['out'],batch=lo,turn=turn+1,active=sum(not x for x in done),elapsed=time.time()-started)
    for q,hs,success,ob in zip(order,hist,won,initial):
     result=dict(**q,history=hs,success=success,initial=ob,policy_version=version,seed=req['seed'],turns=len(hs));save(out/(q['id']+'.json'),result);results.append(result)
   completion=dict(timings=timings,concurrency=chunk,actions=actions_generated,input_tokens=input_tokens,output_tokens=output_tokens,tasks=len(results),success=sum(x['success'] for x in results),seconds=time.time()-started,version=version)
   save(out/'completion.json',completion);return completion
  finally:
   pass # Keep the CPU environment service resident across batches.
 def handle(req):
  nonlocal version
  op=req['op']
  if op=='load_snapshot':
   llm.reset_prefix_cache()
   llm.collective_rpc(load_snapshot,args=(req['path'],))
   version=req['version']
   llm.update_weight_version(str(version))
   return dict(version=version)
  elif op=='episodes':return episodes(req)
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
