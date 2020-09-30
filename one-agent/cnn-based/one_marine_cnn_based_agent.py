from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions,units,features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import time
from absl import app

MAPNAME = 'CollectMineralShards'
APM = 0
APM = int(APM / 18.75)
UNLIMIT = 0
VISUALIZE = False
REALTIME = False # 실험 결과 볼 때, True로 해놓고 보는 용도 
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
K_EPOCH       = 5
T_HORIZON     = 60
EPISODES = 10000

players = [sc2_env.Agent(sc2_env.Race.terran)]

interface = features.AgentInterfaceFormat(\
                feature_dimensions = features.Dimensions(\
                screen = SCREEN_SIZE, minimap = MINIMAP_SIZE), use_feature_units = True)

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv_1 = nn.Conv2d(2,4,3,1,padding = 1)
        self.deconv_1 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
        self.pooling = nn.MaxPool2d(2)
        self.value_1 = nn.Linear(16 * 16 * 4,128)
        self.value_2 = nn.Linear(128,1)
    def forward(self,x):
        x = self.conv_1(x)
        x = F.relu(x)
        encoded = self.pooling(x)
        
        x = self.deconv_1(encoded)
        x = x.view(-1,SCREEN_SIZE*SCREEN_SIZE)
        action = F.softmax(x,-1)
        
        value = encoded.view(-1,4 * int(SCREEN_SIZE/2)*int(SCREEN_SIZE/2))
        value = self.value_1(value)
        value = F.relu(value)
        value = self.value_2(value)
        return action,value

class Agent(base_agent.BaseAgent):
    def __init__(self):
        
        super(Agent,self).__init__()
        self.network = Network()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.data = []
    def step(self,obs):
        super(Agent,self).step(obs)
        if obs.first(): 
            #첫 step일 때, marine 두기중 하나 random으로 선택
            control_marine = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Marine][0]
            return actions.FUNCTIONS.select_point("select",(control_marine.x,control_marine.y))
        elif obs.observation.control_groups[CONTROL_GROUP_SET][0] == 0:
            #두번째 step일 때, 선택된 marine을 부대지정함
            #사실 현재 꼭필요는 없지만 이후 두 marine 모두 이용하는 version에서 필요함
            return actions.FUNCTIONS.select_control_group([CONTROL_GROUP_SET], [MARINE_GROUP_ORDER])
        #환경 전처리
        state = get_state(obs)
        screen = torch.tensor(state).unsqueeze(0).float()
        #action probability와 value 얻음
        action_prob,value = self.network(screen)
        action_dist = Categorical(action_prob)
        #action sampling
        action_coords = action_dist.sample().reshape(-1,1)
        #action -> x,y coordinates로 바꿈
        y,x = torch.cat((action_coords // SCREEN_SIZE, action_coords % SCREEN_SIZE), dim=1)[0]
        #실제 environment에 들어갈 action function call
        action = actions.FunctionCall(MOVE_SCREEN,[NOT_QUEUED,[x.item(),y.item()]])
        #현재 preprocessed state, real action(environment에 들어갈), flatten action, flatten action probabilty
        return state,[action],action_coords[0][0].item(),action_prob[0][action_coords[0][0].item()]
    
    def put_data(self,transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train(self):
        if len(self.data) == 0:
            print("done train error")
            return False
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        for i in range(K_EPOCH):
            pi,v = self.network(s)
            td_target = r + GAMMA * self.network(s_prime)[1] * done_mask
            delta = td_target - v
            delta = delta.detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = GAMMA * LMBDA * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantage
            loss = -torch.min(surr1, surr2).mean() + F.smooth_l1_loss(v , td_target.detach().float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return True
def get_state(obs):
    state_1 = np.expand_dims(np.array(obs.observation.feature_screen.player_relative),0)
    state_2 = np.expand_dims(np.array(obs.observation.feature_screen.selected),0)
    state = np.concatenate([state_1,state_2],0)
    return state
def main(args):
    agent = Agent()
    #agent.network.load_state_dict(torch.load('./model_weights/one_marine_mineralshards_mdp_cnn_500'))
    try:
        episode = 0
        with sc2_env.SC2Env(map_name=MAPNAME, players=players, \
                    agent_interface_format=interface, \
                    step_mul=APM, game_steps_per_episode=UNLIMIT, \
                    visualize=VISUALIZE, realtime=REALTIME) as env:
            while True:
                if episode > EPISODES:
                    break
                episode += 1
                agent.setup(env.observation_spec(), env.action_spec())
                timestep = env.reset()
                agent.reset()
                done = False
                
                while not done:
                    for t in range(T_HORIZON):
                        action_info = agent.step(timestep[0])
                        if len(action_info) == 2:
                            #첫 두스텝은 action info length가 2
                            action = [action_info]
                        else:
                            #첫 두스텝 이후 스텝은 action info length가 5
                            state,action,action_coords,action_prob = action_info
                            #print("action_prob : ",action_prob.item())
                        #이번 스텝만의 reward를 구하려면 다음 step에서 얻는 mineral에서 지금 step에서 가지고 있는 mineral을 빼면됨
                        reward = - timestep[0].observation.player.minerals 
                        timestep = env.step(action)
                        reward += timestep[0].observation.player.minerals
                        if timestep[0].last():
                            done = True
                        if (len(action_info) > 2) :
                            #첫 두스텝 이후 스텝은 action info length가 5
                            next_state = get_state(timestep[0])
                            agent.put_data((state, action_coords, reward/100.0, next_state, action_prob.item(), done))
                        if done == True:
                            break
                    trained = agent.train()
                    if trained:
                        print('trained test : ',list(agent.network.parameters())[0][0][0][0])
                    else:
                        print('train failed')
                    
                if (episode % 50 == 0) and (episode != 0):
                    torch.save(agent.network.state_dict(), './model_weights/one_marine_mineralshards_mdp_cnn_'+str(episode))
    except KeyboardInterrupt:
        pass


app.run(main)