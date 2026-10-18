"""Persistent teacher: scores sampled student states (final hidden states + action-token log-probs)."""
import os,time,json,gc
from pathlib import Path
from common import *
from batching import batches,inputs
ROLE=os.environ.get('ALF_ROLE','teacher')
cache=ROOT/'runtime_cache'/'triton'/ROLE;cache.mkdir(parents=True,exist_ok=True);os.environ['TRITON_CACHE_DIR']=str(cache)

def main():
 from core import setup
 torch=setup()
 import torch.nn.functional as F
 from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
 model=Qwen3_5ForConditionalGeneration.from_pretrained(CFG['teacher'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(CFG['teacher'],local_files_only=True)
 student_tok=AutoTokenizer.from_pretrained(CFG['student'],local_files_only=True)
 assert tok.get_vocab()==student_tok.get_vocab()
 head=model.get_output_embeddings()
 save(ROOT/(ROLE+'_ready.json'),dict(time=time.time()))
 def score(req):
  samples=json.loads(Path(req['manifest']).read_text());out=Path(req['out']);out.mkdir(parents=True,exist_ok=True)
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time();done=0;shapes=[]
  try:
   with torch.no_grad():
    for group in batches(samples,req['microbatch'],req.get('token_budget',32000)):
     x=inputs(group,tok.pad_token_id,torch);h=model.model.language_model(**x).last_hidden_state
     shapes.append(list(x['input_ids'].shape))
     for i,s in enumerate(group):
      n=len(s['tokens']);off=len(s['prompt'])-1;hs=h[i,off:off+n].contiguous()
      log=F.log_softmax(head(hs).float(),dim=-1).gather(1,torch.tensor(s['tokens'],device='cuda')[:,None]).squeeze(1)
      p=out/(s['key']+'.pt');tmp=p.with_suffix('.tmp');torch.save(dict(hidden=hs.cpu(),logprobs=log.cpu()),tmp);tmp.replace(p)
      done+=1
      del hs,log
     del h,x
     status(ROLE,stage='scoring',microbatch=req['microbatch'],done=done,total=len(samples))
   torch.cuda.synchronize()
   result=dict(ok=True,seconds=time.time()-start,states=done,peak_gib=torch.cuda.max_memory_allocated()/2**30,shapes=shapes,microbatch=req['microbatch'])
  except torch.OutOfMemoryError as error:
   result=dict(ok=False,oom=True,error=str(error),microbatch=req['microbatch'],done=done)
  save(out/'metrics.json',result);return result
 def handle(req):
  if req['op']=='shutdown':return dict(ok=True)
  result=score(req);gc.collect();torch.cuda.empty_cache();return result
 serve(ROLE,handle)
if __name__=='__main__':guarded(main,'teacher')
