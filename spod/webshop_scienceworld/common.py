"""Shared durable file-RPC helpers between the trainer, teacher and rollout processes of one run."""
import os,json,time,uuid,traceback
from pathlib import Path
from worker import save
CODE=Path(__file__).resolve().parent
ROOT=Path(os.environ['WS_ROOT'])
CFG=json.loads((ROOT/'config.json').read_text())
def status(role,**kw):save(ROOT/(role+'_status.json'),dict(updated=time.time(),**kw))
def check():
 if (ROOT/'failed.json').exists():raise RuntimeError((ROOT/'failed.json').read_text())
def wait(p,timeout=7200):
 p=Path(p);start=time.time()
 while not p.exists():
  check()
  if time.time()-start>timeout:raise TimeoutError(str(p))
  time.sleep(.3)
 return json.loads(p.read_text())
def rpc(role,op,**kw):
 tag=f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
 p=ROOT/'rpc'/role/(tag+'.json');save(p,dict(op=op,**kw))
 v=wait(p.with_suffix('.done'))
 if 'error' in v:raise RuntimeError(v['error'])
 return v
def serve(role,handler):
 q=ROOT/'rpc'/role;q.mkdir(parents=True,exist_ok=True)
 while not (ROOT/'complete.json').exists():
  check();pending=[p for p in sorted(q.glob('*.json')) if not p.with_suffix('.done').exists()]
  if not pending:time.sleep(.3);continue
  p=pending[0];req=json.loads(p.read_text());status(role,stage=req['op'],request=p.name)
  try:result=handler(req);save(p.with_suffix('.done'),result or {'ok':True})
  except BaseException:
   error=traceback.format_exc();save(p.with_suffix('.done'),dict(error=error));raise
  if req['op']=='shutdown':break
 status(role,stage='complete')
def guarded(fn,role):
 import signal
 def interrupted(sig,frame):raise KeyboardInterrupt(f'Signal {sig}')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:fn()
 except BaseException:
  save(ROOT/'failed.json',dict(role=role,error=traceback.format_exc(),time=time.time()));raise
