"""CPU ALFWorld process. Agent-facing replies exclude expert/hidden state."""
import os,sys,json,contextlib,yaml
from pathlib import Path
R=Path(__file__).resolve().parent
assert os.environ.get('ALFWORLD_DATA'),'Set ALFWORLD_DATA to the ALFWorld data directory (scripts/setup_alfworld.sh)'
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
cfg=yaml.safe_load((R/'alfworld/base_config.yaml').read_text());cfg['general']['use_cuda']=False;cfg['dagger']['training']['max_nb_steps_per_episode']=50
class ExplicitGamesEnv(AlfredTWEnv):
 def __init__(self,config,games):
  self.explicit_games=list(games)
  super().__init__(config,train_eval='eval_in_distribution')
 def collect_game_files(self,verbose=False):
  self.game_files=list(self.explicit_games);self.num_games=len(self.game_files)
world=None
for line in sys.stdin:
 try:
  req=json.loads(line)
  with contextlib.redirect_stdout(sys.stderr):
   if req['op']=='reset':
    if world is not None:world.close()
    budget=int(req.get('max_steps',50));assert 1<=budget<=200
    cfg['dagger']['training']['max_nb_steps_per_episode']=budget
    cfg['rl']['training']['max_nb_steps_per_episode']=budget
    base=ExplicitGamesEnv(cfg,req['games'])
    world=base.init_env(batch_size=len(req['games']));world.seed(req.get('seed',20260916));obs,info=world.reset()
    result=dict(obs=list(obs),commands=[list(x) for x in info['admissible_commands']],games=list(info['extra.gamefile']),won=[False]*len(obs),dones=[False]*len(obs))
   elif req['op']=='step':
    obs,score,done,info=world.step(req['actions']);result=dict(obs=list(obs),commands=[list(x) for x in info['admissible_commands']],won=[bool(x) for x in info['won']],dones=[bool(x) for x in done],scores=[float(x) for x in score])
   elif req['op']=='close':
    if world is not None:world.close()
    result=dict(closed=True)
   else:raise ValueError(req['op'])
  print(json.dumps(result),flush=True)
 except Exception:
  import traceback
  print(json.dumps(dict(error=traceback.format_exc())),flush=True)
