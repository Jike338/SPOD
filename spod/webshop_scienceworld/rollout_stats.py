"""Per-family rollout success counts written each update (logging only)."""
from collections import Counter
from pathlib import Path
from worker import save
def save_rollout_counts(root,update,eps):
 ns=Counter(e['family'] for e in eps);ks=Counter(e['family'] for e in eps if e['success'])
 save(Path(root)/'updates'/f'{update:03}'/'rollout_family_success.json',dict(update=update,rollout_policy_update=update-1,families={f:dict(n=n,successes=ks[f],success=ks[f]/n) for f,n in ns.items()},mean_score=sum(e['score'] for e in eps)/len(eps)))
