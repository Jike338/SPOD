"""Persistent teacher: scores sampled student states with its native prompt (final hidden states + action log-probs)."""
from common import *
from pipeline_io import teacher_prompt
def main():
 import torch
 import torch.nn.functional as F
 from transformers import AutoModelForImageTextToText,AutoTokenizer
 torch.set_num_threads(4)
 model=AutoModelForImageTextToText.from_pretrained(CFG['teacher'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();model.requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(CFG['teacher'],local_files_only=True)
 student_tok=AutoTokenizer.from_pretrained(CFG['student'],local_files_only=True)
 assert tok.get_vocab()==student_tok.get_vocab(), 'Teacher/student vocabulary differs'
 cap=model.config.get_text_config().final_logit_softcapping
 save(ROOT/'teacher_ready.json',dict(time=time.time()))
 def handle(req):
  if req['op']=='shutdown':return {}
  assert req['op']=='score';samples=json.loads(Path(req['manifest']).read_text())
  with torch.inference_mode():
   for j,s in enumerate(sorted(samples,key=lambda x:len(x['prompt'])+len(x['tokens']))):
    target=Path(s['teacher_file'])
    if target.exists():continue
    prompt=teacher_prompt(tok,student_tok,s)
    n=len(s['tokens']);off=len(prompt)-1;ids=torch.tensor([prompt+s['tokens']],device='cuda')
    h=model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0,off:off+n]
    logits=model.get_output_embeddings()(h)
    if cap:logits=torch.tanh(logits/cap)*cap
    lp=F.log_softmax(logits.float(),-1);selected=lp.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).squeeze(1)
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp');torch.save(dict(hidden=h.cpu(),logprobs=selected.cpu(),softcap=cap),tmp);tmp.replace(target)
    if j%20==0:status('teacher',stage='score',done=j+1,total=len(samples))
    del h,logits,lp,selected,ids
  return dict(states=len(samples))
 serve('teacher',handle)
if __name__=='__main__':guarded(main,'teacher')
