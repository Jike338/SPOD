"""Protocol functions: sampling seeds, learning-rate schedule, episode summaries, paired bootstrap."""
import hashlib, math, random, statistics

EVAL_SEEDS = [20260916, 20260917, 20260918]

def request_seed(base, task_id, turn):
    return int.from_bytes(hashlib.sha256(f'{base}|{task_id}|{turn}'.encode()).digest()[:8], 'little') % (2**31 - 1)

def learning_rate(update, cfg):
    total = int(cfg.get('schedule_updates', cfg['end_updates'])); warm = int(cfg.get('warmup_updates', 3))
    assert cfg['end_updates'] <= total and 1 <= update <= total and 0 <= warm <= total
    initial = float(cfg['lr']); final = float(cfg.get('final_lr', initial))
    assert initial > 0 and final >= 0
    if warm and update <= warm:
        return initial * update / warm
    mode = cfg.get('schedule', 'constant')
    progress = (update - warm) / max(1, total - warm)
    if mode == 'constant': return initial
    if mode == 'linear': return initial + (final - initial) * progress
    if mode == 'cosine': return final + (initial - final) * (1 + math.cos(math.pi * progress)) / 2
    raise ValueError(mode)

def length_bin(turns):
    assert 0 <= turns <= 50
    if turns == 0: return '0'
    low = ((turns - 1) // 10) * 10 + 1
    return f'{low}-{low + 9}'

def summarize_episodes(episodes):
    ids = [x['id'] for x in episodes]; assert len(ids) == len(set(ids))
    out = dict(tasks=len(episodes), success=sum(bool(x['success']) for x in episodes), by_task_type={}, by_observed_length={})
    for ep in episodes:
        family = ep.get('family') or ep['game'].split('/')[-3].split('-')[0]
        for table, key in ((out['by_task_type'], family), (out['by_observed_length'], length_bin(ep['turns']))):
            row = table.setdefault(key, dict(tasks=0, success=0, failure=0, total_turns=0))
            row['tasks'] += 1; row['success'] += int(ep['success']); row['failure'] += int(not ep['success']); row['total_turns'] += ep['turns']
    out['episode_lengths'] = {x['id']:dict(turns=x['turns'],success=x['success'],family=x.get('family')) for x in episodes}
    return out

def mean_sd(values):
    return dict(values=values, mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else 0.)

def paired_bootstrap(a, b, repeats=5000, seed=20261001):
    """Pair identical task IDs, average eval draws within each task, then resample tasks."""
    assert set(a) == set(b) and a
    ids = sorted(a); diff = [a[k] - b[k] for k in ids]; n = len(diff)
    rng = random.Random(seed)
    samples = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats))
    return dict(tasks=n, delta_success_tasks=sum(diff), delta_rate=sum(diff)/n,
                ci95_rate=[samples[int(.025*repeats)],samples[min(repeats-1,int(.975*repeats))]],
                bootstrap_replicates=repeats, bootstrap_seed=seed)
