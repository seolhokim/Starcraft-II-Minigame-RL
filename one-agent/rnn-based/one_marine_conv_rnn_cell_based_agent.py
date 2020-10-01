from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions,units,features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import time
import copy
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

CNN_LSTM_CHANNEL_NUM = 4
LEARNING_RATE = 0.001
GAMMA         = 0.98
LMBDA         = 0.95
EPS_CLIP      = 0.1
K_EPOCH       = 10
T_HORIZON     = 50
EPISODES = 10000

players = [sc2_env.Agent(sc2_env.Race.terran)]

interface = features.AgentInterfaceFormat(\
                feature_dimensions = features.Dimensions(\
                screen = SCREEN_SIZE, minimap = MINIMAP_SIZE), use_feature_units = True)
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv_1 = nn.Conv2d(in_channels = 2, out_channels = 4, kernel_size = 3, stride = 1,padding = 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.conv_lstm = ConvLSTMCell(4,CNN_LSTM_CHANNEL_NUM,(3,3),True)
        #self.conv_3 = nn.Conv2d(in_channels = CNN_LSTM_CHANNEL_NUM, out_channels = 1, kernel_size = 3, stride = 1,padding = 1)
        self.deconv_1 = nn.ConvTranspose2d(CNN_LSTM_CHANNEL_NUM, 1, 3, stride=2, padding=1, output_padding=1)
        self.linear_1 = nn.Linear(4*int(SCREEN_SIZE/2 * SCREEN_SIZE/2),128)
        self.linear_2 = nn.Linear(128,1)
    def forward(self,x,hidden_state):
        x = F.relu(self.conv_1(x))
        x = self.maxpooling(x)
        encoded,c = self.conv_lstm(x,hidden_state)
        h = encoded.clone().detach()#copy.deepcopy(encoded)
        x = self.deconv_1(F.relu(encoded))
        x = x.view(-1, SCREEN_SIZE * SCREEN_SIZE)
        action = F.softmax(x,-1)
        value = F.relu(self.linear_1(encoded.view(-1,4*int(SCREEN_SIZE/2 * SCREEN_SIZE/2))))
        value = self.linear_2(value)
        return action,value,(h,c.detach())

class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent,self).__init__()
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
        #if sum([x.order_length for x in obs.observation.feature_units if x.is_selected == 1]) == 1:
        #    return actions.FUNCTIONS.no_op()
        
        state = get_state(obs)
        screen = torch.tensor(state).unsqueeze(0).float()
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
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst, h_out_lst
             
    def train(self):
        if len(self.data) == 0:
            print("done train error")
            return False
        s, a, r, s_prime, done_mask, prob_a, h_in_lst, h_out_lst= self.make_batch()
        first_hidden  = (h_in_lst[0][0], h_in_lst[0][1])
        second_hidden = (h_out_lst[0][0], h_out_lst[0][1])
        for i in range(K_EPOCH):
            pi_lst = []
            v_lst = []
            next_value_lst = []
            for batch_idx in range(s.size(0)):
                pi,v, (h1_in, h2_in) = self.network(s[batch_idx].unsqueeze(0),first_hidden)
                _, next_value, (h1_out, h2_out) = self.network(s_prime[batch_idx].unsqueeze(0),second_hidden)
                pi_lst.append(pi)
                v_lst.append(v)
                next_value_lst.append(next_value)
                first_hidden  = (h1_in, h2_in)
                ####first_hidden  = (h1_in, h2_in)
                second_hidden = (h1_out, h2_out)
                ####second_hidden = (h1_out, h2_out)
                #first_hidden  = (h_in_lst[batch_idx][0].detach(), h_in_lst[batch_idx][1].detach())
                #second_hidden = (h_out_lst[batch_idx][0].detach(), h_out_lst[batch_idx][1].detach())
            pi = torch.stack(pi_lst,0).squeeze(1)
            v = torch.stack(v_lst,0).squeeze(1)
            next_value = torch.stack(next_value_lst,0).squeeze(1)
            td_target = r + GAMMA * next_value * done_mask
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
            #print("loss 1(up) : ",-torch.min(surr1, surr2).mean())
            #print("loss 2(down) : ",F.smooth_l1_loss(v , td_target.detach().float()))
            print("ratio : ",ratio.mean())
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)#retain_graph=True
            self.optimizer.step()
        return True
def get_state(obs):
    state_1 = np.expand_dims(np.array(obs.observation.feature_screen.player_relative),0)
    state_2 = np.expand_dims(np.array(obs.observation.feature_screen.selected),0)
    state = np.concatenate([state_1,state_2],0)
    return state
def main(args):
    agent = Agent()
    #agent.network.load_state_dict(torch.load('./model_weights/one_marine_mineralshards_150'))
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
                
                h_out = (torch.zeros(1, CNN_LSTM_CHANNEL_NUM, int(SCREEN_SIZE/2), int(SCREEN_SIZE/2)),
                torch.zeros(1, CNN_LSTM_CHANNEL_NUM, int(SCREEN_SIZE/2), int(SCREEN_SIZE/2)))
                while not done:
                    for t in range(T_HORIZON):
                        h_in = h_out
                        action_info = agent.step(timestep[0],h_in)
                        if len(action_info) == 2:
                            action = [action_info]
                        else:
                            state,action,action_coords,action_prob,h_out = action_info
                            #print("action : ",action_coords)

                        reward = - timestep[0].observation.player.minerals
                        timestep = env.step(action)
                        reward += timestep[0].observation.player.minerals
                        if timestep[0].last():
                            done = True
                            
                        if (len(action_info) > 2) :
                            next_state = get_state(timestep[0])
                            #print('reward : ',reward)
                            agent.put_data((state, action_coords, reward/100.0, next_state, action_prob.item(),h_in, h_out, done))
                        if done == True:
                            break
                    trained = agent.train()
                    if trained:
                        print('trained test : ',list(agent.network.parameters())[0][0][0][0])
                    else:
                        print('train failed')
                    
                if (episode % 50 == 0) and (episode != 0):
                    torch.save(agent.network.state_dict(), './model_weights/one_marine_mineralshards_rnn_'+str(episode))
    except KeyboardInterrupt:
        pass


app.run(main)