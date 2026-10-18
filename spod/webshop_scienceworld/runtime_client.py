"""Owned rollout subprocess with bounded startup, RPC and cleanup."""
import os, sys, json, time, signal, subprocess, uuid, atexit
from pathlib import Path
from worker import save
CODE = Path(__file__).resolve().parent
class Runtime:
 def __init__(self, root, cfg, gpu=None):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  assert not (self.root/'rollout0_ready.json').exists(), 'Refuse stale runtime reuse'
  save(self.root/'config.json',cfg)
  env=dict(os.environ,WS_ROOT=str(self.root),WS_ROLE='rollout0')
  if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
  self.log=(self.root/'rollout.log').open('a');self.p=subprocess.Popen([sys.executable,'-u',str(CODE/'rollout.py')],env=env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
  save(self.root/'process.json',dict(pid=self.p.pid,time=time.time()));atexit.register(self.close)
  try:self.wait(self.root/'rollout0_ready.json',1800)
  except BaseException:self.close();raise
 def wait(self, path, seconds=14400):
  end=time.time()+seconds
  while not path.exists():
   if self.p.poll() is not None:raise RuntimeError(f'Rollout exited {self.p.returncode}: {self.root}')
   if (self.root/'failed.json').exists():raise RuntimeError((self.root/'failed.json').read_text())
   if time.time()>end:raise TimeoutError(str(path))
   time.sleep(.3)
  v=json.loads(path.read_text());assert 'error' not in v,v;return v
 def request(self,op,**kw):
  p=self.root/'rpc/rollout0'/(uuid.uuid4().hex+'.json');save(p,dict(op=op,**kw));return self.wait(p.with_suffix('.done'))
 def close(self):
  if self.p.poll() is None:
   os.killpg(self.p.pid,signal.SIGTERM)
   try:self.p.wait(timeout=45)
   except subprocess.TimeoutExpired:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
  self.log.close()
 def __enter__(self):return self
 def __exit__(self,typ,*args):
  try:
   if typ is None and self.p.poll() is None:self.request('shutdown');assert self.p.wait(timeout=120)==0
  finally:self.close()
def episodes(runtime, rows, out, seed, version=0, temperature=.6, concurrency=72):
 # Per-request seeds make results independent of batching; concurrency only changes speed.
 m=runtime.request('episodes',version=version,rows=rows,out=out,seed=seed,concurrency=concurrency,temperature=temperature)
 eps=[json.loads((runtime.root/out/(q['id']+'.json')).read_text()) for q in rows]
 assert len(eps)==len(rows) and m['success']==sum(e['success'] for e in eps)
 from fixed_protocol import request_seed
 for e in eps:
  assert e['turns']==len(e['history'])
  for t,h in enumerate(e['history']):assert h['raw']['request_seed']==request_seed(seed,e['id'],t)
 return eps,m
