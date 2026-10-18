"""Two-GPU full-parameter SPOD training on WebShop / ScienceWorld."""
import os,json,time,random,math,gc,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import *
from fixed_protocol import learning_rate
from runtime_client import Runtime,episodes
from evaluation import evaluate_runtime,record_curve
from rollout_stats import save_rollout_counts
from disk_snapshots import stream_bf16
def student_class(path):
 from transformers import AutoConfig,AutoModelForImageTextToText,AutoModelForCausalLM
 return AutoModelForImageTextToText if 'vision_config' in AutoConfig.from_pretrained(path,local_files_only=True).to_dict() else AutoModelForCausalLM
def main():
 import torch
 import torch.nn.functional as F
 from transformers import AutoTokenizer
 from safetensors import safe_open
 assert not (ROOT/'complete.json').exists() and not list((ROOT/'updates').glob('*/completion.json')),'No implicit restart'
 torch.set_num_threads(8);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
 tok=AutoTokenizer.from_pretrained(CFG['student'],local_files_only=True)
 model=student_class(CFG['student']).from_pretrained(CFG['student'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
 for name,p in model.named_parameters():
  text='language_model' in name or name.startswith('lm_head');p.requires_grad_(text)
  if text:p.data=p.data.float()
 params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=0.,foreach=False)
 for p in params:opt.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 teacher=Path(CFG['teacher'])
 if (teacher/'model.safetensors.index.json').exists():index=json.loads((teacher/'model.safetensors.index.json').read_text())['weight_map']
 else:
  with safe_open(str(teacher/'model.safetensors'),framework='pt',device='cpu') as f:index={k:'model.safetensors' for k in f.keys()}
 key='lm_head.weight' if 'lm_head.weight' in index else next(k for k in index if k.endswith('language_model.embed_tokens.weight'))
 with safe_open(str(teacher/index[key]),framework='pt',device='cpu') as f:w=f.get_tensor(key)
 head=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=w.dtype);head.weight=torch.nn.Parameter(w,requires_grad=False)
 save(ROOT/'teacher_head.json',dict(key=key,shape=list(w.shape)))
 wait(ROOT/'teacher_ready.json');rows=json.loads(Path(CFG['train_manifest']).read_text());assert len(rows)>=72*CFG['end_updates']
 TAU=float(CFG['sg_tau'])
 def move(device):
  model.to(device);head.to(device)
  if not CFG.get('adam_cpu_offload'):
   for st in opt.state.values():
    for k,v in st.items():
     if torch.is_tensor(v) and k!='step':st[k]=v.to(device)
  gc.collect();torch.cuda.empty_cache()
 scratch=Path(os.environ['WS_SCRATCH'])/ROOT.name/'rollout_weights'
 if scratch.parent.exists():shutil.rmtree(scratch.parent)  # start from an empty scratch directory
 scratch.parent.mkdir(parents=True,exist_ok=True)
 if not (ROOT/'rollout_weights').exists():(ROOT/'rollout_weights').symlink_to(scratch,target_is_directory=True)
 def export(path):
  path=Path(path);tmp=path.with_name(path.name+'.writing')
  if tmp.exists():shutil.rmtree(tmp)
  tmp.mkdir(parents=True);stream_bf16(model.state_dict(),tmp/'model.safetensors')
  model.config.save_pretrained(tmp);model.generation_config.save_pretrained(tmp);tok.save_pretrained(tmp)
  for extra in ['preprocessor_config.json','chat_template.jinja','processor_config.json','video_preprocessor_config.json']:
   if (Path(CFG['student'])/extra).exists():shutil.copy(Path(CFG['student'])/extra,tmp/extra)
  if path.exists():shutil.rmtree(path)
  tmp.rename(path)
 def runtime(update,out):
  move('cpu');cfg=dict(CFG,eval_model=str(scratch) if update>=1 else CFG['student'],policy_version=update)
  return Runtime(ROOT/'rollout_phases'/out,cfg)
 def evaluate(update):
  status('train',stage='evaluation',update=update)
  with runtime(update,f'eval_{update:03}') as rt:
   result=evaluate_runtime(rt,ROOT/'evaluation'/f'{update:03}',CFG['eval_manifest'],version=update,seeds=CFG['eval_seeds'])
  record_curve(ROOT,update,result)
 evaluate(0)
 pool=ThreadPoolExecutor(max_workers=1)
 for update in range(1,CFG['end_updates']+1):
  tick=time.time();out=ROOT/'updates'/f'{update:03}';out.mkdir(parents=True,exist_ok=True)
  hidden=scratch.parent/f'teacher_{update:03}';hidden.mkdir(parents=True,exist_ok=False);(out/'teacher').symlink_to(hidden,target_is_directory=True)
  selected=rows[(update-1)*72:update*72]
  status('train',stage='rollout',update=update)
  with runtime(update-1,f'update_{update:03}') as rt:eps,roll=episodes(rt,selected,'episodes',CFG['seed']+update-1,update-1)
  save_rollout_counts(ROOT,update,eps)
  states=[];rng=random.Random(CFG['seed']+update-1)
  for ep in eps:
   usable=[h for h in ep['history']]
   chosen=rng.sample(usable,min(8,len(usable)));assert chosen
   for h in chosen:
    raw=h['raw'];k=f'{len(states):04}'
    states.append(dict(key=k,task_id=ep['id'],family=ep['family'],episode_success=bool(ep['success']),turn=h['turn'],prompt=raw['prompt_token_ids'],messages=raw['messages'],tokens=raw['output_token_ids'],weight=1./(len(selected)*len(chosen)),teacher_file=str(out/'teacher'/(k+'.pt'))))
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
     lq=F.log_softmax(logits.float(),-1);err=float((lq.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).squeeze(1)-th['logprobs']).abs().max());assert err<5e-3,err;maxerr=max(maxerr,err)
     teacher_seq=float(th['logprobs'].float().sum())
     del th
    if lq.shape[-1]!=lp.shape[-1]:
     m=min(lq.shape[-1],lp.shape[-1]);lq=lq[:,:m];lp=lp[:,:m]
    kl=(lq.exp()*(lq-lp)).sum()/n
    # SPOD: on a successful episode, drop the teacher term exactly where it would lower the student's action
    # (q_teacher(a) < p_student(a)) and apply the one-sided floor max(0, log tau - log p(a))/n instead.
    gated=False;fl=None;student_seq=None
    if s['episode_success']:
     student_seq=lp.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).sum()
     if teacher_seq<float(student_seq.detach()):
      gated=True;fl=(math.log(TAU)-student_seq).clamp_min(0.)/n
    obj=fl if gated else kl
    loss=obj*s['weight']
   assert torch.isfinite(loss);total+=float(loss.detach());components.append(dict(key=s['key'],family=s['family'],weight=s['weight'],kl=float(kl.detach()),success=s['episode_success'],gated=gated,floor=float(fl.detach()) if gated else 0.,student_seq_logp=float(student_seq.detach()) if student_seq is not None else None,teacher_seq_logp=teacher_seq));loss.backward();del result,lp,lq,logits,kl,loss,ids,obj,fl,student_seq
   if j%20==0:status('train',stage='backward',update=update,done=j+1,total=len(states),loss_so_far=total,seconds=time.time()-st)
  future.result();norm=float(torch.nn.utils.clip_grad_norm_(params,CFG['gradient_clip']));assert math.isfinite(norm)
  torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/2**30
  if CFG.get('adam_cpu_offload'):
   move('cpu');opt.step();opt.zero_grad(set_to_none=True)
  else:
   opt.step();opt.zero_grad(set_to_none=True);move('cpu')
  save(out/'components.json',components)
  fam={}
  for c in components:
   d=fam.setdefault(c['family'],dict(teacher=0.,states=0,gated=0));d['teacher']+=c['weight']*c['kl'];d['states']+=1;d['gated']+=int(c.get('gated',False))
  rec=dict(update=update,lr=rate,loss=total,success_states=sum(1 for c in components if c.get('success')),gated_states=sum(1 for c in components if c.get('gated')),floor_active=sum(1 for c in components if c.get('floor',0)>0),loss_by_family=fam,gradient_norm=norm,states=len(states),teacher_error=maxerr,rollout=roll,seconds=time.time()-tick,peak_gib=peak,time=time.time());save(out/'completion.json',rec);status('train',stage='update_complete',**{k:v for k,v in rec.items() if k!='rollout'})
  export(scratch)
  shutil.rmtree(hidden);(out/'teacher').unlink()
  if update%CFG['eval_every']==0:evaluate(update)
  if update in CFG.get('save_model_updates',[]):export(ROOT/'models'/f'u{update:03}')
  if CFG.get('rolling_save_every') and (update%CFG['rolling_save_every']==0 or update==CFG['end_updates']):
   export(ROOT/'models'/'latest');save(ROOT/'models'/'latest_update.json',dict(update=update,time=time.time(),bf16_weights_only=True,optimizer_saved=False))
 rpc('teacher','shutdown');pool.shutdown()
 save(ROOT/'complete.json',dict(time=time.time(),updates=CFG['end_updates']))
if __name__=='__main__':guarded(main,'train')
