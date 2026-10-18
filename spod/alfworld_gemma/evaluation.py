"""Three draws per checkpoint, paired task bootstrap and length/family summaries."""
import json,os
from pathlib import Path
from fixed_protocol import EVAL_SEEDS, summarize_episodes, mean_sd, paired_bootstrap
from runtime_client import episodes
from worker import save
CODE=Path(__file__).resolve().parent

def evaluate_runtime(runtime, destination, version=0, seeds=EVAL_SEEDS, temperature=.6, label='eval'):
 all_eps=[];scores=[];details=[]
 for i,seed in enumerate(seeds):
  joined=[];splits={}
  for split in ('seen','unseen'):
   rows=json.loads((Path(os.environ['ALF_MANIFEST_DIR'])/f'manifest_{split}.json').read_text())
   eps,metrics=episodes(runtime,rows,f'{label}_{i}_{split}',seed,version,temperature)
   joined+=eps;splits[split]=summarize_episodes(eps)
   save(Path(destination)/f'{label}_{i}_{split}.json',dict(seed=seed,metrics=metrics,summary=splits[split],episodes_path=str(runtime.root/f'{label}_{i}_{split}')))
  summary=summarize_episodes(joined)
  details.append(dict(seed=seed,summary=summary,splits=splits));all_eps.append(joined);scores.append(summary['success'])
 means={e['id']:sum(int(next(x for x in eps if x['id']==e['id'])['success']) for eps in all_eps)/len(all_eps) for e in all_eps[0]}
 result=dict(version=version,temperature=temperature,seeds=seeds,scores=mean_sd(scores),details=details,task_success_means=means)
 save(Path(destination)/(label+'.json'),result);return result

def record_curve(root,update,result):
 root=Path(root);p=root/'learning_curve.json';curve=json.loads(p.read_text()) if p.exists() else {}
 if '0' in curve:result['paired_vs_initial']=paired_bootstrap(result['task_success_means'],curve['0']['task_success_means'])
 curve[str(update)]=result
 save(p,curve)
 # A standalone vector chart; bands are +/-1 evaluation-seed SD, never training-seed SD.
 import html
 xs=sorted(map(int,curve));vals=[curve[str(x)]['scores'] for x in xs]
 def x(v):return 55+v/50*670
 def y(v):return 325-v/274*280
 pts=' '.join(f'{x(a):.1f},{y(b["mean"]):.1f}' for a,b in zip(xs,vals))
 lines=''.join(f'<line x1="{x(a)}" x2="{x(a)}" y1="{y(b["mean"]-b["sd"])}" y2="{y(b["mean"]+b["sd"])}" stroke="#64748b"/><circle cx="{x(a)}" cy="{y(b["mean"])}" r="3" fill="#0369a1"/>' for a,b in zip(xs,vals))
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="390" viewBox="0 0 800 390"><rect width="800" height="390" fill="white"/><text x="55" y="24" font-size="17">{html.escape(root.name)}: mean success / 274, ±1 eval-seed SD</text><path d="M55 40V325H730" stroke="black" fill="none"/><polyline points="{pts}" fill="none" stroke="#0369a1"/>{lines}<text x="365" y="370">Training update (0–50)</text><text x="12" y="50">274</text><text x="30" y="325">0</text></svg>'
 (root/'learning_curve.svg').write_text(svg)
