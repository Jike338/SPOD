"""Evaluation draws over the fixed test manifest (WebShop or ScienceWorld); per-family success and mean score."""
import os,json
from pathlib import Path
from fixed_protocol import summarize_episodes, mean_sd
from runtime_client import episodes
from worker import save
def evaluate_runtime(runtime, destination, manifest, version=0, seeds=None, temperature=.6, label='eval'):
 rows=json.loads(Path(manifest).read_text());scores=[];details=[];per_task={}
 for i,seed in enumerate(seeds):
  eps,metrics=episodes(runtime,rows,f'{label}_{i}',seed,version,temperature,concurrency=min(len(rows),int(os.environ.get('EVAL_MAX_CONC',len(rows)))))  # optional cap on concurrent episodes
  summary=summarize_episodes(eps);details.append(dict(seed=seed,summary=summary,metrics=metrics));scores.append(summary['success'])
  for e in eps:per_task.setdefault(e['id'],[]).append(int(e['success']))
 result=dict(version=version,tasks=len(rows),seeds=seeds,scores=mean_sd(scores),details=details,task_success_means={k:sum(v)/len(v) for k,v in per_task.items()})
 save(Path(destination)/(label+'.json'),result);return result
def record_curve(root,update,result):
 root=Path(root);p=root/'learning_curve.json';curve=json.loads(p.read_text()) if p.exists() else {}
 fam={}
 for d in result['details']:
  for f,v in d['summary']['by_task_type'].items():
   x=fam.setdefault(f,dict(tasks=0,success=0,score=0.));x['tasks']+=v['tasks'];x['success']+=v['success'];x['score']+=v['score']
 curve[str(update)]=dict(scores=result['scores'],tasks=result['tasks'],success_rate=result['scores']['mean']/result['tasks'],mean_score=sum(d['summary']['mean_score'] for d in result['details'])/len(result['details']),families={f:dict(success_rate=v['success']/v['tasks'],mean_score=v['score']/v['tasks'],tasks=v['tasks']) for f,v in fam.items()})
 save(p,curve)
