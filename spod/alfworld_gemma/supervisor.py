"""Own and clean up exactly the two Gemma workers launched for this run."""
import os,sys,time,subprocess,signal,traceback
from pathlib import Path
from worker import save
R=Path(__file__).resolve().parent;root=Path(sys.argv[1]).resolve();name=root.name;assert name and all(c.isalnum() or c in '_-' for c in name)
children=[]
def cleanup():
 for p in children:
  if p.poll() is None:
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 for p in children:
  try:p.wait(timeout=60)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def interrupted(sig,frame):raise KeyboardInterrupt(f'Signal {sig}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
 gpus=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',');assert len(gpus)==2
 for script,gpu in [('teacher.py',gpus[1]),('train.py',gpus[0])]:
  role=script.split('.')[0]
  with (root/(role+'.log')).open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(R/script)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,ALF_RUN_DIR=str(root),CUDA_VISIBLE_DEVICES=gpu))
  children.append(p)
 save(root/'processes.json',dict(pids=[p.pid for p in children],gpus=gpus,time=time.time()))
 while any(p.poll() is None for p in children):
  if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Worker failed: '+str([p.poll() for p in children]))
  if (root/'failed.json').exists():raise RuntimeError((root/'failed.json').read_text())
  time.sleep(3)
 assert (root/'complete.json').exists()
except BaseException:save(root/'supervisor_failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
finally:cleanup()
