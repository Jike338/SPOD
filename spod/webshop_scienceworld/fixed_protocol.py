"""Protocol functions: sampling seeds, learning-rate schedule, episode summaries."""
import hashlib, math, statistics
EVAL_SEEDS=[20260916,20260917,20260918]
def request_seed(base, task_id, turn):
    return int.from_bytes(hashlib.sha256(f'{base}|{task_id}|{turn}'.encode()).digest()[:8], 'little') % (2**31 - 1)
def learning_rate(update, cfg):
    total=int(cfg.get('schedule_updates',cfg['end_updates']));warm=int(cfg.get('warmup_updates',3))
    initial=float(cfg['lr']);final=float(cfg.get('final_lr',initial))
    if warm and update<=warm:return initial*update/warm
    mode=cfg.get('schedule','constant');progress=(update-warm)/max(1,total-warm)
    if mode=='constant':return initial
    if mode=='linear':return initial+(final-initial)*progress
    if mode=='cosine':return final+(initial-final)*(1+math.cos(math.pi*progress))/2
    raise ValueError(mode)
def summarize_episodes(episodes):
    ids=[x['id'] for x in episodes];assert len(ids)==len(set(ids))
    out=dict(tasks=len(episodes),success=sum(bool(x['success']) for x in episodes),mean_score=sum(x.get('score',0.) for x in episodes)/max(1,len(episodes)),by_task_type={})
    for ep in episodes:
        row=out['by_task_type'].setdefault(ep['family'],dict(tasks=0,success=0,score=0.,total_turns=0))
        row['tasks']+=1;row['success']+=int(ep['success']);row['score']+=ep.get('score',0.);row['total_turns']+=ep['turns']
    out['episode_lengths']={x['id']:dict(turns=x['turns'],success=x['success'],score=x.get('score',0.),family=x['family']) for x in episodes}
    return out
def mean_sd(values):
    return dict(values=values,mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else 0.)
