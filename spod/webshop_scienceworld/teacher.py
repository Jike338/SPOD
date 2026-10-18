"""Teacher scoring: native teacher prompt, final hidden states and action log-probs saved per sampled state."""
from common import *
from pipeline_io import teacher_prompt
def load_model(path):
 import torch
 from transformers import AutoConfig,AutoModelForImageTextToText,AutoModelForCausalLM
 cls=AutoModelForImageTextToText if 'vision_config' in AutoConfig.from_pretrained(path,local_files_only=True).to_dict() else AutoModelForCausalLM
 return cls.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,device_map='cuda:0')
def backbone(model):
 return model.model if hasattr(model,'model') else model.base_model
def main():
 import torch
 import torch.nn.functional as F
 from transformers import AutoTokenizer
 torch.set_num_threads(4)
 model=load_model(CFG['teacher']).eval();model.requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(CFG['teacher'],local_files_only=True)
 student_tok=AutoTokenizer.from_pretrained(CFG['student'],local_files_only=True)
 assert tok.get_vocab()==student_tok.get_vocab(),'Teacher/student vocabulary differs'
 cap=getattr(model.config.get_text_config(),'final_logit_softcapping',None)
 save(ROOT/'teacher_ready.json',dict(time=time.time(),softcap=cap,cls=type(model).__name__))
 def score(s):
  """Teacher hidden states and action-token log-probs for one sampled state (native teacher prompt)."""
  p=teacher_prompt(tok,student_tok,s);ids=torch.tensor([p+s['tokens']],dtype=torch.long,device='cuda')
  n=len(s['tokens']);off=len(p)-1
  h=backbone(model)(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0,off:off+n]
  logits=model.get_output_embeddings()(h)
  if cap:logits=torch.tanh(logits/cap)*cap
  lp=F.log_softmax(logits.float(),-1)
  return h,lp.gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).squeeze(1)
 def handle(req):
  if req['op']=='shutdown':return {}
  assert req['op']=='score';samples=json.loads(Path(req['manifest']).read_text())
  todo=[s for s in sorted(samples,key=lambda x:len(x['prompt'])+len(x['tokens'])) if not Path(s['teacher_file']).exists()]
  with torch.inference_mode():
   for j,s in enumerate(todo):
    h,selected=score(s)
    target=Path(s['teacher_file']);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp');torch.save(dict(hidden=h.cpu(),logprobs=selected.cpu(),softcap=cap),tmp);tmp.replace(target)
    status('teacher',stage='score',done=j+1,total=len(todo))
  return dict(states=len(samples))
 serve('teacher',handle)
if __name__=='__main__':guarded(main,'teacher')
