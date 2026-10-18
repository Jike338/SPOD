"""ScienceWorld environment process: JSON lines over stdin/stdout (reset / step / close), one simulator per episode slot."""
import os,sys,json,traceback,contextlib
from concurrent.futures import ThreadPoolExecutor
from scienceworld.constants import JAR_PATH
from scienceworld import ScienceWorldEnv
envs=[];pool=ThreadPoolExecutor(max_workers=int(os.environ['SW_CONCURRENCY']))
def state(e,obs,info,done=False,valid=True):
 return dict(observation=obs,inventory=info['inv'],score=max(0,info['score'])/100.,success=float(e.server.getScore())>=1.,done=bool(done),valid=valid)
def reset_one(pair):
 e,row=pair;e.load(row['task'],row['variation'],'');obs,info=e.reset();v=state(e,obs,info);v['task_description']=e.get_task_description();return v
def step_one(pair):
 e,a=pair
 if a is None:return None
 obs,reward,done,info=e.step(a);auto=0
 while obs.startswith('Ambiguous request:') and auto<3:
  obs,reward,done,info=e.step('0');auto+=1   # resolve simulator disambiguation prompts with the first option
 rejected=obs.startswith(('No known action matches','Unknown action',"I don't understand"))
 return state(e,obs,info,done,not rejected)
def handle(q):
 if q['op']=='reset':
  while len(envs)<len(q['rows']):envs.append(ScienceWorldEnv('',serverPath=str(JAR_PATH),envStepLimit=100))
  return dict(states=list(pool.map(reset_one,zip(envs,q['rows']))))
 if q['op']=='step':return dict(states=list(pool.map(step_one,zip(envs,q['actions']))))
 if q['op']=='close':
  for e in envs:e.close()
  pool.shutdown();return {}
 raise ValueError(q)
for line in sys.stdin:
 try:
  q=json.loads(line)
  with contextlib.redirect_stdout(sys.stderr):v=handle(q)
  print(json.dumps(v),flush=True)
  if q['op']=='close':break
 except BaseException:print(json.dumps(dict(error=traceback.format_exc())),flush=True)
