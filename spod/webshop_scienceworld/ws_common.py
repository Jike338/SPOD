"""Shared WebShop setup (verl-agent/GiGPO defaults: use_small=True, human_goals=False -> synthetic goals over 1,000 products; test goals 0-499, train 500+): deterministic goal list, task families, available commands (Python 3.8)."""
import os, sys, random
WS_HOME = os.environ['WS_HOME']  # WebShop checkout prepared by scripts/setup_webshop.sh
sys.path.insert(0, WS_HOME)
GOAL_SEED = 20260916  # price_upper sampling in get_human_goals is unseeded upstream; fix it once for all processes
SEARCH = 'search[<your query>]'

def make_server():
    random.seed(GOAL_SEED)
    from web_agent_site.envs.web_agent_text_env import SimServer
    server = SimServer('http://127.0.0.1:3000', os.path.join(WS_HOME, 'data/items_shuffle_1000.json'),
                       filter_goals=None, limit_goals=-1, num_products=None, human_goals=0, show_attrs=False)
    return server

def commands(env):
    a = env.get_available_actions()
    if a['has_search_bar']:
        return [SEARCH]
    return ['click[%s]' % c for c in a['clickables'] if c != 'search']
