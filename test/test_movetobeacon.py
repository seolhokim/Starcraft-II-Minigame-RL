from pysc2.env import sc2_env
from pysc2.lib import features 
from pysc2.agents import base_agent
from pysc2.lib import actions

from absl import app

import numpy as np

import torch

MAPNAME = 'MoveToBeacon'
APM = 0
APM = int(APM / 18.75)
UNLIMIT = 0
VISUALIZE = False
REALTIME = True
if REALTIME :
    REALTIME_GAME_LOOP_SECONDS = 1
else:
    REALTIME_GAME_LOOP_SECONDS = 22.4
    
SCREEN_SIZE = 32
MINIMAP_SIZE = 16

CONTROL_GROUP_SET = 1

MARINE_GROUP_ORDER = 1

MOVE_SCREEN = 331
NOT_QUEUED = [0]

LEARNING_RATE = 0.001
GAMMA         = 0.98
LMBDA         = 0.95
EPS_CLIP      = 0.1
K_EPOCH       = 10
T_HORIZON     = 1000
EPISODES = 10000
players = [sc2_env.Agent(sc2_env.Race.terran),]

interface = features.AgentInterfaceFormat(\
                feature_dimensions = features.Dimensions(\
                screen = SCREEN_SIZE, minimap = MINIMAP_SIZE), use_feature_units = True)

class Agent(base_agent.BaseAgent):
    def step(self,obs):
        super(Agent,self).step(obs)
        return actions.FUNCTIONS.no_op()
    


def main(args):
    agent = Agent()
    try:
        episode = 0
        with sc2_env.SC2Env(map_name = MAPNAME, players = players,\
                agent_interface_format = interface,\
                step_mul = APM, game_steps_per_episode = UNLIMIT,\
                visualize = VISUALIZE, realtime = REALTIME) as env:
            
            while True:
                if episode > EPISODES:
                    break
                episode += 1
                agent.setup(env.observation_spec(), env.action_spec())
                timestep = env.reset()
                agent.reset()
                done = False
                while not done :
                    for t in range(T_HORIZON):
                        step_actions = [agent.step(timestep[0])]
                        
                        timestep = env.step(step_actions)
                        reward = timestep[0].reward
                        if timestep[0].last():
                            done = True
                            break
    except KeyboardInterrupt:
        pass
app.run(main)