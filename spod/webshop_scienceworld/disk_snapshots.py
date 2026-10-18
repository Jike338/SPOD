"""Streaming BF16 safetensors writer (one tensor conversion at a time)."""
import hashlib, json, os, struct
from pathlib import Path

DTYPES={'torch.bfloat16':('BF16',2),'torch.float32':('F32',4),'torch.float64':('F64',8),'torch.float16':('F16',2),'torch.int64':('I64',8),'torch.int32':('I32',4),'torch.int16':('I16',2),'torch.int8':('I8',1),'torch.uint8':('U8',1),'torch.bool':('BOOL',1)}

def drop_file_cache(path):
 if hasattr(os,'posix_fadvise'):
  with open(path,'rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)

def stream_bf16(state,path,guard=None):
 """No full converted state dict, model deepcopy, or FP32 model clone."""
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 header={'__metadata__':{'format':'pt'}};offset=0
 for name,t in sorted(state.items()):
  dtype=torch.bfloat16 if t.is_floating_point() else t.dtype
  code,width=DTYPES[str(dtype)];end=offset+t.numel()*width
  header[name]=dict(dtype=code,shape=list(t.shape),data_offsets=[offset,end]);offset=end
 raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
 tmp=path.with_suffix(path.suffix+'.writing');assert not path.exists() and not tmp.exists()
 h=hashlib.sha256()
 with tmp.open('xb') as f:
  f.write(struct.pack('<Q',len(raw)));f.write(raw)
  for name,t in sorted(state.items()):
   if guard:guard.check(extra_bytes=t.numel()*(2 if t.is_floating_point() else t.element_size()),stage='snapshot_tensor:'+name)
   c=t.detach().to(device='cpu',dtype=torch.bfloat16 if t.is_floating_point() else t.dtype).contiguous()
   h.update(name.encode());h.update(str(tuple(c.shape)).encode())
   data=c.reshape(-1).view(torch.uint8).numpy();view=memoryview(data)
   for start in range(0,len(view),8*1024**2):
    chunk=view[start:start+8*1024**2];f.write(chunk);h.update(chunk)
   del view,data,c
  f.flush();os.fsync(f.fileno())
 assert tmp.stat().st_size==8+len(raw)+offset
 # Validate the header and each tensor's extent with the official mmap reader.
 from safetensors import safe_open
 with safe_open(str(tmp),framework='pt',device='cpu') as sf:
  assert set(sf.keys())==set(state)
  for name in state:assert list(sf.get_slice(name).get_shape())==header[name]['shape']
 tmp.rename(path);drop_file_cache(path)
 return h.hexdigest()
