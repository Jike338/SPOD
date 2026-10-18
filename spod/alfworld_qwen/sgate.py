"""SPOD loss pieces.

On states of SUCCESSFUL episodes where the teacher term would LOWER the student's action (exact gate from the forward-KL
gradient: q_teacher(a) < p_student(a)), the teacher term is dropped and replaced by the one-sided floor
max(0, log tau - log p_student(a)) / n_tokens. All other states use the teacher full-vocabulary forward KL.
Probabilities are full-vocabulary action-sequence probabilities (sum of token log-probs, including EOS), measured with the
HF BF16 student and the recorded teacher scores.
"""
import math
from worker import save


def floor_loss(seq_logp, n, tau):
    # One-sided: zero gradient once the successful action is at least tau-likely.
    return (math.log(tau) - seq_logp).relu() / n


def mark_success(states, episodes):
    for s in states:
        s['success'] = bool(episodes[s['task_id']]['success'])
    return states


def record(root, cfg, states, values, update):
    gated = [v for v in values.values() if v.get('gated')]
    save(root / 'updates' / f'{update:03}' / 'sgate.json',
         dict(update=update, tau=cfg['sg_tau'], states=len(values), success_states=sum(1 for s in states if s['success']),
              gated_states=len(gated), floor_active=sum(1 for v in gated if v.get('floor', 0) > 0),
              mean_teacher_p_gated=(sum(math.exp(v['teacher_seq_logp']) for v in gated) / len(gated)) if gated else None,
              mean_student_p_gated=(sum(math.exp(v['student_seq_logp']) for v in gated) / len(gated)) if gated else None))
