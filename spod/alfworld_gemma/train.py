"""Two-GPU full-parameter SPOD training on ALFWorld (Gemma student)."""
import os,json,time,random,math,gc,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import *
from fixed_protocol import learning_rate
from runtime_client import Runtime,episodes
from evaluation import evaluate_runtime,record_curve
from bookkeeping import save_rollout_counts, checkpoint_export
from disk_snapshots import stream_bf16

def main():
 import torch
 import torch.nn.functional as F
 from transformers import Gemma4ForConditionalGeneration,AutoTokenizer
 from safetensors import safe_open
 assert not (ROOT/'complete.json').exists() and not list((ROOT/'updates').glob('*/completion.json')),'No implicit restart'
 torch.set_num_threads(4);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
 tok=AutoTokenizer.from_pretrained(CFG['student'],local_files_only=True)
 model=Gemma4ForConditionalGeneration.from_pretrained(CFG['student'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
 for name,p in model.named_parameters():
  text='language_model' in name or name.startswith('lm_head');p.requires_grad_(text)
  if text:p.data=p.data.float()
 params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=0.,foreach=False)
 for p in params:opt.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
 model.cpu();torch.cuda.empty_cache();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 teacher=Path(CFG['teacher'])
 if (teacher/'model.safetensors.index.json').exists():index=json.loads((teacher/'model.safetensors.index.json').read_text())['weight_map']
 else:
  with safe_open(str(teacher/'model.safetensors'),framework='pt',device='cpu') as f:index={k:'model.safetensors' for k in f.keys()}
 key='lm_head.weight' if 'lm_head.weight' in index else next(k for k in index if k.endswith('language_model.embed_tokens.weight'))
 with safe_open(str(teacher/index[key]),framework='pt',device='cpu') as f:w=f.get_tensor(key)
 head=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=w.dtype);head.weight=torch.nn.Parameter(w,requires_grad=False)
 wait(ROOT/'teacher_ready.json');rows=json.loads((ROOT/'manifest_train.json').read_text())
 def move(device):
  model.to(device);head.to(device)
  if not CFG.get('adam_cpu_offload'):
   for st in opt.state.values():
    for k,v in st.items():
     if torch.is_tensor(v) and k!='step':st[k]=v.to(device)
   del v
  else:
   assert all(v.device.type=='cpu' for st in opt.state.values() for v in st.values() if torch.is_tensor(v))

  gc.collect();torch.cuda.empty_cache()
 scratch=Path(os.environ['TMPDIR'])/ROOT.name/'rollout_weights';scratch.parent.mkdir(parents=True,exist_ok=True)
 (ROOT/'rollout_weights').symlink_to(scratch,target_is_directory=True)
 def export(path):
  if Path(path)==ROOT/'rollout_weights':path=scratch
  path=Path(path)
  tmp=path.with_name(path.name+'.writing');assert not tmp.exists();tmp.mkdir(parents=True)
  stream_bf16(model.state_dict(),tmp/'model.safetensors')
  model.config.save_pretrained(tmp);model.generation_config.save_pretrained(tmp);tok.save_pretrained(tmp)
  if path.exists():shutil.rmtree(path)
  tmp.rename(path)
 def runtime(update,out):
  move('cpu');cfg=dict(CFG,eval_model=str(ROOT/'rollout_weights') if update>=1 else CFG['student'],rollout_memory=.90,policy_version=update)
  rt=Runtime(ROOT/'rollout_phases'/out,cfg)
  return rt

 def evaluate(update):
  status('train',stage='evaluation',update=update)
  with runtime(update,f'eval_{update:03}') as rt:
   result=evaluate_runtime(rt,ROOT/'evaluation'/f'{update:03}',version=update,seeds=CFG['eval_seeds'])
  record_curve(ROOT,update,result)
 evaluate(0)
 pool=ThreadPoolExecutor(max_workers=1)
 for update in range(1,CFG['end_updates']+1):
  tick=time.time();out=ROOT/'updates'/f'{update:03}';out.mkdir(parents=True,exist_ok=True)
  hidden=scratch.parent/f'teacher_{update:03}';hidden.mkdir(parents=True,exist_ok=False);(out/'teacher').symlink_to(hidden,target_is_directory=True)
  selected=rows[(update-1)*72:min(update*72,3553)]
  status('train',stage='rollout',update=update)
  with runtime(update-1,f'update_{update:03}') as rt:eps,roll=episodes(rt,selected,'episodes',CFG['seed']+update-1,update-1)
  save_rollout_counts(ROOT,update,eps)
  states=[];rng=random.Random(CFG['seed']+update-1)
  for ep in eps:
   chosen=rng.sample(ep['history'],min(8,len(ep['history'])));assert chosen
   for h in chosen:
    raw=h['raw'];key=f'{len(states):04}'
    states.append(dict(key=key,task_id=ep['id'],family=ep['family'],episode_success=bool(ep['success']),turn=h['turn'],prompt=raw['prompt_token_ids'],messages=raw['messages'],tokens=raw['output_token_ids'],weight=1./(len(selected)*len(chosen)),teacher_file=str(out/'teacher'/(key+'.pt'))))
  assert math.isclose(sum(s['weight'] for s in states),1.,abs_tol=1e-12)
  save(out/'samples.json',states)
  future=pool.submit(rpc,'teacher','score',manifest=str(out/'samples.json'))
  move('cuda');model.train();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();total=0.;st=time.time();maxerr=0.;components=[]
  rate=learning_rate(update,CFG)
  for group in opt.param_groups:group['lr']=rate
  for j,s in enumerate(sorted(states,key=lambda x:len(x['prompt'])+len(x['tokens']))):
   while not Path(s['teacher_file']).exists():
    check()
    if future.done():future.result()
    time.sleep(.1)
   n=len(s['tokens']);off=len(s['prompt'])-1;ids=torch.tensor([s['prompt']+s['tokens']],device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    result=model(input_ids=ids,use_cache=False,logits_to_keep=torch.arange(off,off+n,device='cuda'));lp=F.log_softmax(result.logits[0].float(),-1)
    with torch.no_grad():
     th=torch.load(s['teacher_file'],map_location='cuda',weights_only=True);logits=head(th['hidden']);tc=th['softcap']
     if tc:logits=torch.tanh(logits/tc)*tc
     lq=F.log_softmax(logits.float(),-1);err=float((lq.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).squeeze(1)-th['logprobs']).abs().max());assert err<5e-4,err;maxerr=max(maxerr,err)
     teacher_seq=float(th['logprobs'].float().sum())
     del th
    kl=(lq.exp()*(lq-lp)).sum()/n
    # SPOD: on a successful episode, drop the teacher term exactly where it would lower the student's action
    # (q_teacher(a) < p_student(a)) and apply the one-sided floor max(0, log tau - log p(a))/n instead.
    fkl_val=float(kl.detach());gated=False;floor_val=0.;student_seq_val=None
    if s['episode_success']:
     student_seq=lp.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).sum();student_seq_val=float(student_seq.detach())
     if teacher_seq<student_seq_val:
      gated=True;kl=(math.log(float(CFG['sg_tau']))-student_seq).clamp_min(0.)/n;floor_val=float(kl.detach())
     del student_seq
    loss=kl*s['weight']
   assert torch.isfinite(loss);total+=float(loss.detach());components.append(dict(key=s['key'],family=s['family'],weight=s['weight'],kl=float(kl.detach()),gated=gated,fkl=fkl_val,floor=floor_val,success=s['episode_success'],student_seq_logp=student_seq_val,teacher_seq_logp=teacher_seq));loss.backward();del result,lp,lq,logits,kl,loss,ids
   if j%20==0:status('train',stage='backward',update=update,done=j+1,total=len(states),loss_so_far=total,seconds=time.time()-st)
  future.result();norm=float(torch.nn.utils.clip_grad_norm_(params,CFG['gradient_clip']));assert math.isfinite(norm) and norm>0
  torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/2**30
  if CFG.get('adam_cpu_offload'):
   status('train',stage='cpu_adam',update=update)
   move('cpu')
   assert all(p.device.type=='cpu' and p.grad is not None and p.grad.device.type=='cpu' for p in params)
   assert all(v.device.type=='cpu' for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
   opt.step();opt.zero_grad(set_to_none=True)
   save(out/'cpu_adam_step.json',dict(time=time.time(),steps=sorted({float(st['step']) for st in opt.state.values()}),parameters=len(params),fp32_parameters=all(p.dtype==torch.float32 for p in params),fp32_moments=all(st['exp_avg'].dtype==st['exp_avg_sq'].dtype==torch.float32 for st in opt.state.values()),device='cpu',weight_decay=0.0,foreach=False))
  else:
   opt.step();opt.zero_grad(set_to_none=True);move('cpu')
  save(out/'components.json',components);assert math.isclose(total,sum(x['weight']*x['kl'] for x in components),rel_tol=1e-6,abs_tol=1e-8)
  assert {p.stem for p in hidden.glob('*.pt')}=={s['key'] for s in states}
  rec=dict(update=update,tasks=min(update*72,3553),objective=CFG['objective'],success_states=sum(1 for c in components if c.get('success')),gated_states=sum(1 for c in components if c.get('gated')),floor_active=sum(1 for c in components if c.get('floor',0)>0),lr=rate,loss=total,gradient_norm=norm,states=len(states),teacher_error=maxerr,rollout=roll,seconds=time.time()-tick,peak_gib=peak,time=time.time());save(out/'completion.json',rec);status('train',stage='update_complete',**rec)
  export(ROOT/'rollout_weights')
  if update in CFG['checkpoint_updates']:
   checkpoint_export(ROOT,update,export)
  shutil.rmtree(hidden);(out/'teacher').unlink()
  if update%CFG['eval_every']==0:evaluate(update)
 rpc('teacher','shutdown');pool.shutdown();shutil.rmtree(scratch);(ROOT/'rollout_weights').unlink()
 retained=sorted(int(p.name[6:]) for p in (ROOT/'checkpoints').glob('update*'));assert retained==CFG['checkpoint_updates']
 save(ROOT/'complete.json',dict(time=time.time(),updates=CFG['end_updates'],retained_checkpoints=retained))
if __name__=='__main__':guarded(main,'train')
