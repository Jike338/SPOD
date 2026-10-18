"""Per-update bookkeeping: rollout success counts and BF16 checkpoint export."""
from pathlib import Path
from collections import Counter
from worker import save

def save_rollout_counts(root,update,eps):
 ns=Counter(e['family'] for e in eps);ks=Counter(e['family'] for e in eps if e['success'])
 save(Path(root)/'updates'/f'{update:03}'/'rollout_family_success.json',dict(update=update,rollout_policy_update=update-1,families={f:dict(n=n,successes=ks[f],success=ks[f]/n) for f,n in ns.items()},episodes=[dict(id=e['id'],family=e['family'],success=bool(e['success'])) for e in eps]))

def checkpoint_export(root,u,export):
 # BF16 weights only (no optimizer state).
 root=Path(root);cp=root/'checkpoints'/f'update{u:03}'
 assert not cp.exists();export(cp)
 save(cp/'retained.json',dict(update=u,weights=True,optimizer=False,weights_bytes=sum(p.stat().st_size for p in cp.glob('*.safetensors'))))
