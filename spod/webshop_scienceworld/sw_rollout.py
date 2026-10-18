"""ScienceWorld rollout worker: vLLM student policy, free-form actions, episodes up to 100 turns, served through the
pipeline's file-RPC 'episodes' op. The simulator runs in sw_env/environment_bridge.py under SCIENCEWORLD_PYTHON."""
import os,json,time,subprocess
from pathlib import Path
from common import *
from fixed_protocol import request_seed
S=Path(__file__).resolve().parent/'sw_env'
import importlib.util
_spec=importlib.util.spec_from_file_location('sw_prompt',str(S/'sw_prompt.py'));swref=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(swref)
ROLE=os.environ.get('WS_ROLE','rollout0')
def clean(text):
 t=text.strip().split('\n')[0].strip().strip('`').strip()
 if t.lower().startswith('action:'):t=t[7:].strip()
 return t
def main():
 from transformers import AutoTokenizer, GenerationConfig
 from vllm import LLM,SamplingParams
 status(ROLE,stage='loading')
 model_path=CFG.get('eval_model',CFG['student'])
 tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 try:gen_eos=GenerationConfig.from_pretrained(model_path,local_files_only=True).eos_token_id
 except OSError:gen_eos=None
 gen_eos=[] if gen_eos is None else [gen_eos] if isinstance(gen_eos,int) else list(gen_eos)
 stop=sorted(set(gen_eos+[tok.eos_token_id]+[tok.convert_tokens_to_ids(t) for t in ('<turn|>','<end_of_turn>','<|im_end|>') if t in tok.get_vocab()]))
 llm=LLM(model=model_path,dtype='bfloat16',tensor_parallel_size=CFG.get('tp',1),max_model_len=12000,max_num_seqs=CFG.get('max_num_seqs',256),max_num_batched_tokens=16384,gpu_memory_utilization=CFG.get('rollout_memory',.85),enforce_eager=True,enable_prefix_caching=True,limit_mm_per_prompt={'image':0,'video':0,'audio':0},seed=CFG['seed'])
 version=int(CFG.get('policy_version',0))
 save(ROOT/(ROLE+'_ready.json'),dict(time=time.time(),pid=os.getpid(),stop_ids=stop))
 def episodes(req):
  assert req['version']==version
  rows=req['rows'];out=ROOT/req['out'];out.mkdir(parents=True,exist_ok=True);conc=int(req.get('concurrency',72))
  log=(ROOT/(ROLE+'_environment.log')).open('a')
  bridge=subprocess.Popen([str(swref.ENV_PY),'-u',str(S/'environment_bridge.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAVA_TOOL_OPTIONS='-Xmx1g -XX:-UsePerfData',SW_CONCURRENCY=str(min(conc,len(rows)))),cwd=str(S))
  def ask(q):
   bridge.stdin.write(json.dumps(q)+'\n');bridge.stdin.flush();line=bridge.stdout.readline();assert line,'Environment bridge exited';v=json.loads(line);assert 'error' not in v,v;return v
  results=[];started=time.time()
  try:
   for lo in range(0,len(rows),conc):
    group=rows[lo:lo+conc];states=ask(dict(op='reset',rows=group))['states'];task=[s['task_description'] for s in states];hist=[[] for _ in group];done=[False]*len(group)
    for t in range(CFG['max_steps']):
     check();active=[i for i in range(len(group)) if not done[i]]
     if not active:break
     prompts=[];params=[];raw={};actions=[None]*len(group)
     for i in active:
      messages,ids,trim=swref.prompt(tok,task[i],states[i],hist[i]);rs=request_seed(req['seed'],group[i]['id'],t)
      raw[i]=dict(prompt_token_ids=ids,messages=messages,request_seed=rs,base_seed=req['seed'],history_pairs_trimmed=trim,policy_version=version)
      prompts.append(dict(prompt_token_ids=ids));params.append(SamplingParams(temperature=req['temperature'],top_p=.95 if req['temperature'] else 1.0,top_k=20 if req['temperature'] else -1,max_tokens=CFG['response_tokens'],seed=rs,stop_token_ids=stop,skip_special_tokens=True))
     for i,g in zip(active,llm.generate(prompts,params,use_tqdm=False)):
      o=g.outputs[0];actions[i]=clean(o.text);raw[i].update(text=o.text,output_token_ids=list(o.token_ids),finish=o.finish_reason)
     nxt=ask(dict(op='step',actions=actions))['states']
     for i in active:
      states[i]=nxt[i];a=nxt[i]
      hist[i].append(dict(turn=t+1,action=actions[i],observation=a['observation'],valid=a.get('valid',True),raw=raw[i]))
      done[i]=bool(a['done'] or a['success'])
     status(ROLE,stage='episodes',out=req['out'],batch=lo,turn=t+1,active=sum(not d for d in done),elapsed=time.time()-started)
    for i,q in enumerate(group):
     ep=dict(**q,history=hist[i],success=bool(states[i]['success']),score=max(0.,float(states[i].get('score') or 0)),turns=len(hist[i]),invalid_actions=sum(not h['valid'] for h in hist[i]),policy_version=version,seed=req['seed'])
     save(out/(q['id']+'.json'),ep);results.append(ep)
  finally:
   try:ask(dict(op='close'))
   except Exception:pass
   bridge.terminate()
   try:bridge.wait(timeout=20)
   except subprocess.TimeoutExpired:bridge.kill()
   log.close()
  comp=dict(tasks=len(results),success=sum(r['success'] for r in results),mean_score=sum(r['score'] for r in results)/max(1,len(results)),responses=sum(r['turns'] for r in results),invalid=sum(r['invalid_actions'] for r in results),seconds=time.time()-started,version=version)
  save(out/'completion.json',comp);return comp
 def handle(req):
  if req['op']=='episodes':return episodes(req)
  if req['op']=='shutdown':return dict(ok=True)
  raise ValueError(req['op'])
 try:serve(ROLE,handle)
 finally:
  try:llm.llm_engine.engine_core.shutdown()
  except Exception:pass
