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
REALTIME = False
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
players = [sc2_env.Agent(sc2_env.Race.terran)]

interface = features.AgentInterfaceFormat(\
                feature_dimensions = features.Dimensions(\
                screen = SCREEN_SIZE, minimap = MINIMAP_SIZE), use_feature_units = True)


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.maxpooling = nn.MaxPool2d(2)
        self.conv_1 = nn.Conv2d(2,4,3,1,padding = 1)
        self.lstm = nn.LSTM(4 * int(SCREEN_SIZE) * int(SCREEN_SIZE),4 * int((SCREEN_SIZE) * (SCREEN_SIZE)))
        
        self.conv_2 = nn.Conv2d(4,1,3,1,padding = 1)
        self.deconv_1 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
        #self.deconv_1_1 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
        self.value_1 = nn.Linear(1 * 32*32,128)
        self.value_2 = nn.Linear(128,1)
    def forward(self,x,hidden_state):
        batch_size = x.size(0)
        x = F.relu(self.conv_1(x))
        x = x.view(batch_size,1, 4 * int(SCREEN_SIZE * SCREEN_SIZE))
        x,hidden = self.lstm(x,hidden_state)
        x = x.view(batch_size,4,int(SCREEN_SIZE) , int(SCREEN_SIZE))
        encoded = F.relu(self.conv_2(x))
        #encoded = self.pooling(x)
        
        #x = F.relu(self.deconv_1(encoded))
        #x = (self.deconv_1_1(x))
        x = encoded.view(-1,SCREEN_SIZE*SCREEN_SIZE)
        action = F.softmax(x,-1)
        
        value = encoded.view(-1,1 * int(SCREEN_SIZE)*int(SCREEN_SIZE))
        value = self.value_1(value)
        value = F.relu(value)
        value = self.value_2(value)
        return action,value,hidden

class Agent(base_agent.BaseAgent):
    def __init__(self):
        
        super(Agent,self).__init__()
        if device == 'cuda':
            self.network = Network().cuda()
        else:
            self.network = Network()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.data = []
    def step(self,obs,h_in):
        super(Agent,self).step(obs)
        if obs.first(): 
            control_marine = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Marine][0]
            return actions.FUNCTIONS.select_point("select",(control_marine.x,control_marine.y))
        elif obs.observation.control_groups[CONTROL_GROUP_SET][0] == 0:
            return actions.FUNCTIONS.select_control_group([CONTROL_GROUP_SET], [MARINE_GROUP_ORDER])
        state = get_state(obs)
        screen = torch.tensor(state).unsqueeze(0).float().to(device)
        action_prob,value,h_out = self.network(screen,h_in)
        action_dist = Categorical(action_prob)
        action_coords = action_dist.sample().reshape(-1,1)
        y,x = torch.cat((action_coords // SCREEN_SIZE, action_coords % SCREEN_SIZE), dim=1)[0]
        action = actions.FunctionCall(MOVE_SCREEN,[NOT_QUEUED,[x.item(),y.item()]])
        return state,[action],action_coords[0][0].item(),action_prob[0][action_coords[0][0].item()],h_out
    
    def put_data(self,transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
                                         torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                         torch.tensor(done_lst, dtype=torch.float).to(device), torch.tensor(prob_a_lst).to(device)
        self.data = []
        
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train(self):
        if len(self.data) == 0:
            print("done train error")
            return False
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())
        for i in range(K_EPOCH):
            pi,v,_ = self.network(s,first_hidden)
            td_target = r + GAMMA * self.network(s_prime,second_hidden)[1] * done_mask
            delta = td_target - v
            delta = delta.detach().cpu().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = GAMMA * LMBDA * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantage
            loss = -torch.min(surr1, surr2).mean() + F.smooth_l1_loss(v , td_target.detach().float())
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return True
def get_state(obs):
    state_1 = np.expand_dims(np.array(obs.observation.feature_screen.player_relative),0)
    state_2 = np.expand_dims(np.array(obs.observation.feature_screen.selected),0)
    state = np.concatenate([state_1,state_2],0)
    return state
def main(args):
    agent = Agent()
    #agent.network.load_state_dict(torch.load('./model_weights/one_marine_mineralshards_mdp_rnn_500'))
    try:
        episode = 0
        score = 0
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
                h_out = (torch.zeros(1,1,4 * int(SCREEN_SIZE) * int(SCREEN_SIZE)).to(device),
                torch.zeros(1,1,4 * int(SCREEN_SIZE) * int(SCREEN_SIZE)).to(device))
                while not done:
                    for t in range(T_HORIZON):
                        h_in = h_out
                        action_info = agent.step(timestep[0],h_in)
                        if len(action_info) == 2:
                            action = [action_info]
                        else:
                            state,action,action_coords,action_prob,h_out = action_info
                            #print("action_prob : ",action_prob.item())

                        reward = - timestep[0].observation.player.minerals 
                        timestep = env.step(action)
                        reward += timestep[0].observation.player.minerals
                        score += reward
                        if timestep[0].last():
                            done = True
                        if (len(action_info) > 2) :
                            next_state = get_state(timestep[0])
                            agent.put_data((state, action_coords, reward/100.0, next_state, action_prob.item(),h_in, h_out, done))
                        if done == True:
                            print('score : ',score)
                            score = 0
                            break
                    trained = agent.train()
                    if trained:
                        #print('trained test : ',list(agent.network.parameters())[0][0][0][0])
                        pass
                    else:
                        print('train failed')
                    
                if (episode % 50 == 0) and (episode != 0):
                    torch.save(agent.network.state_dict(), './model_weights/one_marine_mineralshards_mdp_rnn_'+str(episode))
    except KeyboardInterrupt:
        pass


app.run(main)
