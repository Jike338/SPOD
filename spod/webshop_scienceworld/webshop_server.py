"""Persistent WebShop text-environment server (one product load). JSON-lines over a unix socket.
Per connection: a set of episode slots.  ops: reset(games=[goal_idx..]) -> obs/commands/won/dones/scores ; step(actions) ; close.
won = reward==1.0 (strict success); scores = raw WebShop reward in [0,1]."""
import os, sys, json, socketserver, threading, traceback, time
from ws_common import make_server, commands
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
SOCK = sys.argv[1]
t0 = time.time(); SERVER = make_server(); LOCK = threading.Lock()
print(json.dumps(dict(ready=True, goals=len(SERVER.goals), load_seconds=time.time() - t0)), flush=True)

class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        envs, max_steps, steps, finished = [], 50, [], []
        for line in self.rfile:
            try:
                req = json.loads(line)
                with LOCK:
                    if req['op'] == 'reset':
                        max_steps = int(req.get('max_steps', 15))
                        envs = []
                        obs = []
                        for g in req['games']:
                            e = WebAgentTextEnv(observation_mode='text', server=SERVER)
                            o, _ = e.reset(session=int(g)); envs.append(e); obs.append(o)
                        steps = [0] * len(envs); finished = [False] * len(envs)
                        res = dict(obs=obs, commands=[commands(e) for e in envs], games=list(req['games']),
                                   won=[False] * len(envs), dones=[False] * len(envs), scores=[0.0] * len(envs))
                    elif req['op'] == 'step':
                        obs, cmds, won, dones, scores = [], [], [], [], []
                        for i, (e, a) in enumerate(zip(envs, req['actions'])):
                            if finished[i]:
                                obs.append(''); cmds.append([]); won.append(False); dones.append(True); scores.append(0.0); continue
                            o, r, d, _ = e.step(a); steps[i] += 1
                            d = bool(d) or steps[i] >= max_steps
                            finished[i] = d
                            obs.append(o); cmds.append([] if d else commands(e)); won.append(bool(r == 1.0)); dones.append(d); scores.append(float(r))
                        res = dict(obs=obs, commands=cmds, won=won, dones=dones, scores=scores)
                    elif req['op'] == 'close':
                        envs = []; res = dict(closed=True)
                    else:
                        raise ValueError(req['op'])
            except Exception:
                res = dict(error=traceback.format_exc())
            self.wfile.write((json.dumps(res) + '\n').encode()); self.wfile.flush()

class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
if os.path.exists(SOCK): os.unlink(SOCK)
S(SOCK, Handler).serve_forever()
