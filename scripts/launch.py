"""Launch one SPOD training run on two GPUs (student rollouts + trainer on the first GPU, teacher scoring on the second).

usage: python scripts/launch.py configs/qwen_alfworld.json --run-dir runs/qwen_alfworld [--gpus 0,1]

The script resolves the config into an absolute run configuration (local model snapshots, task manifests), writes it to
<run-dir>/config.json and starts the pipeline of the config's package:
  alfworld_qwen          Qwen3.5-2B  <- Qwen3.5-27B   on ALFWorld
  alfworld_gemma         Gemma-4-E4B <- Gemma-4-31B   on ALFWorld
  webshop_scienceworld   either model pair            on WebShop or ScienceWorld
Environment variables (see README):
  ALFWORLD_DATA, ALFWORLD_PYTHON           ALFWorld data directory / interpreter with `alfworld` installed
  WS_HOME, WEBSHOP_PYTHON                  WebShop checkout / interpreter of the WebShop environment
  SCIENCEWORLD_PYTHON                      interpreter with `scienceworld` installed
"""
import argparse, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(value, indent=1)); tmp.replace(path)


def resolve_model(name, revision):
    if Path(name).is_dir():
        return str(Path(name).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=name, revision=revision)


def alfworld_manifest(path):
    data = os.environ.get('ALFWORLD_DATA')
    assert data, 'Set ALFWORLD_DATA to the ALFWorld data directory (scripts/setup_alfworld.sh)'
    rows = json.loads((REPO / path).read_text())
    for r in rows:
        r['game'] = str(Path(data) / r['game'])
        assert Path(r['game']).exists(), f"Missing ALFWorld game file {r['game']}"
    return rows


def supervise(procs, run_dir, driver):
    """Wait for the driver process; stop every child if any worker fails."""
    try:
        while driver.poll() is None:
            if (run_dir / 'failed.json').exists():
                raise RuntimeError((run_dir / 'failed.json').read_text())
            bad = [p for p in procs if p.poll() not in (None, 0)]
            if bad:
                raise RuntimeError(f'Worker exited with code {bad[0].returncode}; see logs in {run_dir}')
            time.sleep(5)
        if driver.returncode != 0:
            raise RuntimeError(f'Run exited with code {driver.returncode}; see logs in {run_dir}')
    finally:
        for p in procs:
            if p.poll() is None:
                try: os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for p in procs:
            try: p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()


def start(cmd, log, env, cwd):
    with open(log, 'a') as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('--run-dir', required=True); ap.add_argument('--gpus', default='0,1')
    args = ap.parse_args()
    spec = json.loads(Path(args.config).read_text())
    gpus = args.gpus.split(','); assert len(gpus) == 2, 'SPOD uses two GPUs: --gpus <student>,<teacher>'
    run_dir = Path(args.run_dir).resolve()
    assert not (run_dir / 'complete.json').exists(), f'{run_dir} already finished'
    assert not (run_dir / 'config.json').exists(), f'{run_dir} already holds a run; use a new --run-dir'
    run_dir.mkdir(parents=True, exist_ok=True)

    package = spec.pop('package'); data = spec.pop('data')
    cfg = dict(spec)
    cfg['student'] = resolve_model(spec['student'], spec.pop('student_revision', None)); cfg.pop('student_revision', None)
    cfg['teacher'] = resolve_model(spec['teacher'], spec.pop('teacher_revision', None)); cfg.pop('teacher_revision', None)
    code = REPO / 'spod' / package
    scratch = Path(os.environ.get('SPOD_SCRATCH', tempfile.gettempdir())) / f'spod_{run_dir.name}_{os.getpid()}'
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(scratch))
    procs = []

    if package == 'alfworld_qwen':
        train = alfworld_manifest(data['train']); cfg['tasks'] = len(train)
        save(run_dir / 'manifest_train.json', train)
        save(run_dir / 'manifest_seen.json', alfworld_manifest(data['seen']))
        save(run_dir / 'manifest_unseen.json', alfworld_manifest(data['unseen']))
        save(run_dir / 'config.json', cfg)
        for role, script, gpu in [('rollout0', 'rollout.py', gpus[0]), ('teacher', 'teacher.py', gpus[1]), ('train', 'train.py', gpus[0])]:
            e = dict(env, ALF_RUN_DIR=str(run_dir), ALF_ROLE=role, CUDA_VISIBLE_DEVICES=gpu)
            procs.append(start([sys.executable, '-u', str(code / script)], run_dir / f'{role}.log', e, code))
        supervise(procs, run_dir, procs[-1])

    elif package == 'alfworld_gemma':
        save(run_dir / 'manifest_train.json', alfworld_manifest(data['train']))
        save(run_dir / 'manifest_seen.json', alfworld_manifest(data['seen']))
        save(run_dir / 'manifest_unseen.json', alfworld_manifest(data['unseen']))
        save(run_dir / 'config.json', cfg)
        e = dict(env, ALF_MANIFEST_DIR=str(run_dir), CUDA_VISIBLE_DEVICES=','.join(gpus))
        procs.append(start([sys.executable, '-u', str(code / 'supervisor.py'), str(run_dir)], run_dir / 'supervisor.log', e, code))
        supervise(procs, run_dir, procs[0])

    elif package == 'webshop_scienceworld':
        cfg['train_manifest'] = str(REPO / data['train']); cfg['eval_manifest'] = str(REPO / data['eval'])
        e = dict(env, WS_SCRATCH=str(scratch), CUDA_VISIBLE_DEVICES=','.join(gpus))
        if cfg.get('env') != 'scienceworld':
            assert os.environ.get('WS_HOME'), 'Set WS_HOME to the WebShop checkout (scripts/setup_webshop.sh)'
            sock = str(Path(tempfile.gettempdir()) / f'spod_ws_{os.getpid()}.sock')  # short path: AF_UNIX limit
            log = run_dir / 'webshop_server.log'
            server = start([os.environ.get('WEBSHOP_PYTHON', sys.executable), '-u', str(code / 'webshop_server.py'), sock], log, env, code)
            procs.append(server)
            while '"ready"' not in (log.read_text() if log.exists() else ''):
                assert server.poll() is None, f'WebShop server exited; see {log}'
                time.sleep(5)
            cfg['ws_socket'] = sock
        save(run_dir / 'config.json', cfg)
        procs.append(start([sys.executable, '-u', str(code / 'supervisor.py'), str(run_dir)], run_dir / 'supervisor.log', e, code))
        supervise(procs, run_dir, procs[-1])
    else:
        raise ValueError(package)
    print(f'Finished: {run_dir}')


if __name__ == '__main__':
    main()
