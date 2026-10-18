"""vLLM student-policy rollouts against the persistent WebShop server."""
import os,json,time,socket
from common import *
from protocol import SYSTEM
from pipeline_io import generation_prompt, extract_action, grammar
from fixed_protocol import request_seed
ROLE=os.environ.get('WS_ROLE','rollout0')
class Env:
 def __init__(self,path):
  self.s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.s.connect(path);self.f=self.s.makefile('rw')
 def ask(self,q):
  self.f.write(json.dumps(q)+'\n');self.f.flush();line=self.f.readline();assert line,'WebShop server closed';v=json.loads(line);assert 'error' not in v,v['error'];return v
def main():
 from transformers import AutoTokenizer, GenerationConfig
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 status(ROLE,stage='loading')
 model_path=CFG.get('eval_model',CFG['student'])
 tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 # Stop tokens: generation-config EOS (if any) plus the tokenizer EOS and the chat end-of-turn token, so an exported
 # checkpoint (which writes its own generation_config.json) parses actions exactly like the base model.
 try:gen_eos=GenerationConfig.from_pretrained(model_path,local_files_only=True).eos_token_id
 except OSError:gen_eos=None
 gen_eos=[] if gen_eos is None else [gen_eos] if isinstance(gen_eos,int) else list(gen_eos)
 stop_ids=sorted(set(gen_eos+[tok.eos_token_id]+[tok.convert_tokens_to_ids(t) for t in ('<|im_end|>','<turn|>','<end_of_turn>') if t in tok.get_vocab()]))
 save(ROOT/(ROLE+'_stop_ids.json'),dict(stop_ids=stop_ids,model=model_path))
 llm=LLM(model=model_path,dtype='bfloat16',tensor_parallel_size=CFG.get('tp',1),max_model_len=16384,max_num_seqs=CFG.get('max_num_seqs',256),max_num_batched_tokens=CFG.get('max_num_batched_tokens',16384),gpu_memory_utilization=CFG.get('rollout_memory',.85),enforce_eager=CFG.get('enforce_eager',True),enable_prefix_caching=True,limit_mm_per_prompt={'image':0,'video':0,'audio':0},seed=CFG['seed'],structured_outputs_config=dict(backend='xgrammar'))
 version=int(CFG.get('policy_version',0))
 save(ROOT/(ROLE+'_ready.json'),dict(time=time.time(),pid=os.getpid()))
 def episodes(req):
  assert req['version']==version,(req['version'],version)
  max_steps=CFG['max_steps'];rows=req['rows'];out=ROOT/req['out'];out.mkdir(parents=True,exist_ok=True)
  env=Env(CFG['ws_socket'])
  results=[];started=time.time();timings=dict(environment=0.,generation=0.,prompt=0.);input_tokens=0;output_tokens=0;chunk=int(req.get('concurrency',72))
  for lo in range(0,len(rows),chunk):
   batch=rows[lo:lo+chunk];tick=time.time()
   state=env.ask(dict(op='reset',games=[q['game'] for q in batch],max_steps=max_steps));timings['environment']+=time.time()-tick
   order=list(batch);assert [q['game'] for q in order]==state['games']
   initial=list(state['obs']);hist=[[] for _ in order];done=[False]*len(order);won=[False]*len(order);score=[0.]*len(order)
   for turn in range(max_steps):
    check();active=[i for i in range(len(order)) if not done[i] and len(hist[i])<max_steps]
    if not active:break
    tick=time.time();prompts=[];params=[];message_batches=[]
    for i in active:
     previous=hist[i]
     while True:
      h='\n'.join('Action: '+x['action']+'\nObservation: '+x['observation'] for x in previous if x['action'] is not None)
      content='Initial observation:\n'+initial[i]+'\nHistory:\n'+h+'\nCurrent observation:\n'+state['obs'][i]+'\nAvailable commands:\n'+'\n'.join(state['commands'][i])
      messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)]
      ids=generation_prompt(tok,messages)
      if len(ids)+CFG['response_tokens']<=CFG['context_limit']:break
      assert previous,'Initial/current observation cannot fit';previous=previous[1:]
     prompts.append(dict(prompt_token_ids=ids));message_batches.append(messages)
     params.append(SamplingParams(temperature=req['temperature'],top_p=.95 if req['temperature'] else 1.0,top_k=20 if req['temperature'] else -1,max_tokens=CFG['response_tokens'],seed=request_seed(req['seed'],order[i]['id'],len(hist[i])),skip_special_tokens=False,structured_outputs=StructuredOutputsParams(**grammar(state['commands'][i]))))
    timings['prompt']+=time.time()-tick;tick=time.time()
    generated=llm.generate(prompts,params,use_tqdm=False);timings['generation']+=time.time()-tick
    actions=['click[back to search]']*len(order);raws={};accepted=[]
    for i,x,pr,messages in zip(active,generated,prompts,message_batches):
     o=x.outputs[0];tt=list(o.token_ids);assert tt
     try:action=extract_action(tok,tt,state['commands'][i],stop_ids)
     except ValueError:action=None
     raws[i]=dict(request_seed=request_seed(req['seed'],order[i]['id'],len(hist[i])),base_seed=req['seed'],text=o.text,messages=messages,prompt_token_ids=pr['prompt_token_ids'],output_token_ids=tt,policy_version=version,finish=o.finish_reason,truncated=o.finish_reason=='length',extracted_action=action,response_tokens=len(tt))
     input_tokens+=len(pr['prompt_token_ids']);output_tokens+=len(tt)
     if action is not None:actions[i]=action;accepted.append(i)
     else:done[i]=True
    tick=time.time();nxt=env.ask(dict(op='step',actions=actions)) if accepted else state;timings['environment']+=time.time()-tick
    for i in active:
     valid=raws[i]['extracted_action'] is not None
     observation=nxt['obs'][i] if valid else state['obs'][i]
     if valid:
      won[i]=bool(nxt['won'][i]);score[i]=max(score[i],float(nxt['scores'][i]));done[i]=done[i] or nxt['dones'][i] or len(hist[i])+1>=max_steps
     hist[i].append(dict(turn=len(hist[i])+1,action=raws[i]['extracted_action'],raw=raws[i],available_commands=state['commands'][i],observation=observation,won=won[i],unclear=not valid))
    # Slots that finished stay finished on the server; only active slots read the new state.
    state=dict(obs=[nxt['obs'][i] if i in accepted else state['obs'][i] for i in range(len(order))],commands=[nxt['commands'][i] if i in accepted else state['commands'][i] for i in range(len(order))]) if accepted else state
    status(ROLE,stage='episodes',version=version,out=req['out'],batch=lo,turn=turn+1,active=sum(not x for x in done),elapsed=time.time()-started)
   for q,hs,success,sc,ob in zip(order,hist,won,score,initial):
    result=dict(**q,history=hs,success=success,score=sc,initial=ob,policy_version=version,seed=req['seed'],turns=len(hs));save(out/(q['id']+'.json'),result);results.append(result)
  env.ask(dict(op='close'))
  completion=dict(unclear_turns=sum(h['unclear'] for x in results for h in x['history']),truncated_turns=sum(h['raw']['truncated'] for x in results for h in x['history']),responses=sum(len(x['history']) for x in results),timings=timings,input_tokens=input_tokens,output_tokens=output_tokens,tasks=len(results),success=sum(x['success'] for x in results),mean_score=sum(x['score'] for x in results)/max(1,len(results)),seconds=time.time()-started,version=version)
  save(out/'completion.json',completion);return completion
 def handle(req):
  if req['op']=='episodes':return episodes(req)
  if req['op']=='shutdown':return dict(ok=True)
  raise ValueError(req['op'])
 try:serve(ROLE,handle)
 finally:llm.llm_engine.engine_core.shutdown()
if __name__=='__main__':
 if CFG.get('env')=='scienceworld':
  import sw_rollout;guarded(sw_rollout.main,'rollout')
 else:guarded(main,'rollout')
