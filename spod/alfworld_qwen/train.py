"""Full-parameter SPOD training on ALFWorld (Qwen student), with live policy synchronization to the vLLM rollout."""
import os,time,json,random,math,gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import *
from batching import batches,inputs
cache=ROOT/'runtime_cache/triton/train';cache.mkdir(parents=True,exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(cache)

def main():
 from core import setup
 torch=setup()
 torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
 wait(ROOT/'rollout0_ready.json')
 import torch.nn.functional as F
 from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
 from safetensors import safe_open
 model=Qwen3_5ForConditionalGeneration.from_pretrained(CFG['student'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
 for name,p in model.named_parameters():
  text=('language_model' in name or name.startswith('lm_head'));p.requires_grad_(text)
  if text:p.data=p.data.float()
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 params=[p for p in model.parameters() if p.requires_grad]
 tok=AutoTokenizer.from_pretrained(CFG['student'],local_files_only=True)
 # Full-vocabulary objectives do not use grammar masks; legal actions are checked by rollout.
 teacher=Path(CFG['teacher']);index=json.loads((teacher/'model.safetensors.index.json').read_text())['weight_map']
 with safe_open(str(teacher/index['lm_head.weight']),framework='pt',device='cpu') as h:w=h.get_tensor('lm_head.weight').cuda()
 head=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=w.dtype);head.weight=torch.nn.Parameter(w,requires_grad=False)
 def optimizer():
  opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=0.,foreach=False)
  # Allocate AdamW state up front.
  for p in params:opt.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
  return opt
 opt=optimizer()
 start_update=0
 from schedule import learning_rate
 from sgate import floor_loss, mark_success, record
 status('train',stage='loaded',resume_update=start_update,text_parameters=sum(p.numel() for p in params),allocated_gib=torch.cuda.memory_allocated()/2**30)

 teachers=['teacher'];nrollouts=1
 for role in teachers:wait(ROOT/(role+'_ready.json'))
 for i in range(nrollouts):wait(ROOT/f'rollout{i}_ready.json')
 def sync(version,count=None):
  from safetensors.torch import save_file
  start=time.time()
  syncdir=ROOT/'sync';syncdir.mkdir(exist_ok=True)
  path=syncdir/'current.safetensors';tmp=syncdir/'current.tmp'
  state={name:param.detach().to(device='cpu',dtype=torch.bfloat16).clone().contiguous() for name,param in model.named_parameters(remove_duplicate=False)}
  save_file(state,str(tmp));tmp.replace(path);del state;gc.collect()
  rpc('rollout0','load_snapshot',path=str(path),version=version)
  return time.time()-start
 def backward(samples,mb):
  model.train();opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time();total=0.;shapes=[];values={}
  try:
   for group in batches(samples,mb,28000):
    for s in group:
     while not Path(s['teacher_file']).exists():check();time.sleep(.05)
    check()
    x=inputs(group,tok.pad_token_id,torch);shapes.append(list(x['input_ids'].shape));loss=None
    with torch.autocast('cuda',dtype=torch.bfloat16):
     h=model.model.language_model(**x).last_hidden_state
     for i,s in enumerate(group):
      n=len(s['tokens']);off=len(s['prompt'])-1
      lp=F.log_softmax(model.get_output_embeddings()(h[i,off:off+n]).float(),dim=-1)
      nll=-lp.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).mean()
      hinge=0.0;extra={}
      with torch.no_grad():
       th=torch.load(s['teacher_file'],weights_only=True,map_location='cuda');lq=F.log_softmax(head(th['hidden']).float(),dim=-1)
       rec=lq.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).squeeze(1)
       err=float((rec-th['logprobs']).abs().max());assert err<5e-4,err
       teacher_seq=float(th['logprobs'].float().sum());del rec,th
      student_seq=-nll*n
      # Exact gate: the FKL term would lower this verified-successful action iff q_teacher(a) < p_student(a).
      gated=s['success'] and teacher_seq<float(student_seq.detach())
      fkl=(lq.exp()*(lq-lp)).sum(-1).mean();del lq
      fl=floor_loss(student_seq,n,CFG['sg_tau']) if gated else None
      kl=fl if gated else fkl
      hinge=float(fl.detach()) if gated else 0.0
      extra=dict(gated=bool(gated),teacher_seq_logp=teacher_seq,student_seq_logp=float(student_seq.detach()),fkl=float(fkl.detach()),floor=hinge,success=s['success'])
      components=dict(loss=float(kl.detach()),nll=float(nll.detach()),hinge=hinge,family=s['family'],weight=s['weight'],key=s['key'],**extra)
      term=kl*s['weight'];loss=term if loss is None else loss+term
      save(ROOT/'components'/f'{current_update:03}'/(s['key']+'.json'),components)
      values[s['key']]=dict(nll=components['nll'],hinge=hinge,loss=components['loss'],**extra);total+=float(term.detach())
      del lp,nll,kl,term,fkl
    assert torch.isfinite(loss);loss.backward();del loss,h,x
    status('train',stage='backward',update=current_update,done=len(values),total=len(samples),loss_so_far=total)
   torch.cuda.synchronize()
   result=dict(ok=True,microbatch=mb,seconds=time.time()-start,loss=total,states=len(samples),peak_gib=torch.cuda.max_memory_allocated()/2**30,shapes=shapes,values=values)
   return result
  except torch.OutOfMemoryError as error:
   return dict(ok=False,oom=True,microbatch=mb,error=str(error))

 rows=json.loads((ROOT/'manifest_train.json').read_text())
 assert len(rows)==CFG['tasks'] and len({q['id'] for q in rows})==len(rows)
 pool=ThreadPoolExecutor(max_workers=4)
 sync(start_update)
 def evaluate(update):
  # Three evaluation seeds; files for the first seed carry no suffix.
  for seed in [CFG['eval_seed'],CFG['eval_seed']+1,CFG['eval_seed']+2]:
   suffix='' if seed==CFG['eval_seed'] else f'_s{seed}'
   for split in ['seen','unseen']:
    n=min(update*CFG["batch_size"],CFG["tasks"])
    path=ROOT/'evaluation'/f'tasks{n}_{split}{suffix}.json'
    if path.exists():continue
    status('train',stage='evaluation',update=update,tasks=n,split=split,seed=seed)
    ev=json.loads((ROOT/f'manifest_{split}.json').read_text())
    metrics=rpc('rollout0','episodes',version=update,rows=ev,out=f'evaluation/tasks{n}_{split}{suffix}',seed=seed,concurrency=CFG['batch_size'])
    assert metrics['tasks']==len(ev)
    save(path,dict(**metrics,accuracy=metrics['success']/metrics['tasks'],training_tasks=n,eval_seed=seed))
 def checkpoint(update,force=False):
  if not force and update not in CFG['milestones'] and update!=CFG['updates']:return 0.
  tick=time.time()
  # One rolling BF16 save (overwritten) plus the final model; no optimizer state.
  export=ROOT/'checkpoints'/(f'final_u{update:03}' if update==CFG['updates'] else 'latest')
  state={n:t.detach().cpu().to(torch.bfloat16) if t.is_floating_point() else t.detach().cpu() for n,t in model.state_dict().items()}
  model.save_pretrained(export,state_dict=state,safe_serialization=True);tok.save_pretrained(export);del state
  save(export/'provenance.json',dict(update=update,config=CFG,optimizer_saved=False,time=time.time(),initialization='base model',objective='SPOD'))
  return time.time()-tick
 if start_update==0 or start_update in CFG['milestones']:evaluate(start_update)
 completed_updates=start_update
 for update in range(start_update,CFG['updates']):
  current_update=update+1
  check();tick=time.time();out=ROOT/'updates'/f'{update+1:03}';out.mkdir(parents=True,exist_ok=True)
  selected=rows[update*CFG['batch_size']:min((update+1)*CFG['batch_size'],CFG['tasks'])];assert len(selected)==min(CFG['batch_size'],CFG['tasks']-update*CFG['batch_size'])
  status('train',stage='rollout',update=update+1)
  roll=rpc('rollout0','episodes',version=update,rows=selected,out=str((out/'episodes').relative_to(ROOT)),seed=CFG['seed']+update,concurrency=CFG['batch_size'])
  roll_wall=time.time()-tick;states=[];rng=random.Random(CFG['seed']+update)
  for q in selected:
   ep=json.loads((out/'episodes'/(q['id']+'.json')).read_text());assert ep['policy_version']==update
   chosen=rng.sample(ep['history'],min(CFG['states_per_task'],len(ep['history'])))
   assert chosen
   for h in chosen:
    raw=h['raw'];assert raw['policy_version']==update and raw['output_token_ids']
    key=f'{len(states):04}'
    states.append(dict(key=key,task_id=q['id'],family=q['family'],turn=h['turn'],prompt=raw['prompt_token_ids'],tokens=raw['output_token_ids'],weight=1/(len(selected)*len(chosen)),teacher_file=str(out/'teacher'/(key+'.pt')),available_commands=h['available_commands'],allowed_tokens=[]))
  assert math.isclose(sum(x['weight'] for x in states),1.)
  episodes={q['id']:json.loads((out/'episodes'/(q['id']+'.json')).read_text()) for q in selected}
  states=mark_success(states,episodes)
  rate=learning_rate(current_update)
  for param_group in opt.param_groups:param_group['lr']=rate
  if (out/'samples.json').exists():assert json.loads((out/'samples.json').read_text())==states,'Refuse stale teacher reuse'
  save(out/'samples.json',states)
  def score_shards():
   st=time.time();futures=[];ordered=sorted(states,key=lambda x:len(x['prompt'])+len(x['tokens']))
   for i,role in enumerate(teachers):
    manifest=out/f'teacher_shard{i}.json';save(manifest,ordered[i::len(teachers)])
    futures.append(pool.submit(rpc,role,'score',manifest=str(manifest),out=str(out/'teacher'),microbatch=1,token_budget=32000))
   result=[f.result() for f in futures]
   return dict(ok=all(x['ok'] for x in result),seconds=time.time()-st,workers=result,states=sum(x.get('states',0) for x in result))
  future=pool.submit(score_shards)
  train=backward(states,1);teacher_result=future.result();assert train['ok'] and teacher_result['ok'],(train,teacher_result)
  record(ROOT,CFG,states,train['values'],current_update)
  norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(norm) and norm>0
  opt.step();opt.zero_grad(set_to_none=True)
  sync_seconds=sync(update+1)
  compute_seconds=time.time()-tick
  ckseconds=checkpoint(update+1)
  summary=dict(lr=rate,sg_tau=CFG['sg_tau'],optimizer_saved=False,update=update+1,tasks=min((update+1)*CFG['batch_size'],CFG['tasks']),loss=train['loss'],gradient_norm=norm,states=len(states),rollout=roll,rollout_wall_seconds=roll_wall,teacher=teacher_result,student=train,sync_seconds=sync_seconds,compute_seconds=compute_seconds,checkpoint_seconds=ckseconds,seconds=time.time()-tick,finished=time.time())
  save(out/'completion.json',summary)
  status('train',stage='update_complete',update=update+1,tasks=summary['tasks'],loss=summary['loss'],seconds=summary['seconds'])
  print('UPDATE_COMPLETE',json.dumps({k:summary[k] for k in ['update','tasks','loss','gradient_norm','compute_seconds','checkpoint_seconds','seconds']}),flush=True)
  completed_updates=update+1
  if completed_updates in CFG['milestones']:evaluate(completed_updates)
 rpc('rollout0','shutdown')
 for role in teachers:rpc(role,'shutdown')
 pool.shutdown()
 save(ROOT/('complete.json' if completed_updates==CFG['updates'] else 'paused.json'),dict(tasks=min(completed_updates*CFG['batch_size'],CFG['tasks']),updates=completed_updates,time=time.time(),planned_updates=CFG['updates']))

if __name__=='__main__':guarded(main,'train')
