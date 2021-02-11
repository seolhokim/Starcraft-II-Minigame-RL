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

import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

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

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.pooling = nn.MaxPool2d(2)
        self.conv_1 = nn.Conv2d(2,4,3,1,padding = 1)
        
        self.conv_2 = nn.Conv2d(4,4,3,1,padding = 1)
        self.conv_lstm = ConvLSTM(input_dim=4,
                 hidden_dim=[4],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
        self.deconv_1 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
        self.value_1 = nn.Linear(int(4 * SCREEN_SIZE/2*SCREEN_SIZE/2),128)
        self.value_2 = nn.Linear(128,1)
    def forward(self,x,hidden_state):
        batch_size = x.size(0)
        x = F.relu(self.conv_1(x))
        x_ = self.pooling(x)
        x = x_.view(batch_size,1, 4 , int(SCREEN_SIZE/2) , int(SCREEN_SIZE/2))
        x,hidden = self.conv_lstm(x)
        x = x[0]
        hidden = hidden[0]
        x = x.view(-1,4,int(SCREEN_SIZE/2), int(SCREEN_SIZE/2))
        encoded = F.relu(self.conv_2(F.relu(x)))
        x = self.deconv_1(encoded)
        x = x.view(-1,SCREEN_SIZE*SCREEN_SIZE)
        action = F.softmax(x,-1)
        
        value = encoded.view(-1,4 * int(SCREEN_SIZE/2)*int(SCREEN_SIZE/2))
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
            #첫 step일 때, marine 두기중 하나 random으로 선택
            control_marine = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Marine][0]
            return actions.FUNCTIONS.select_point("select",(control_marine.x,control_marine.y))
        elif obs.observation.control_groups[CONTROL_GROUP_SET][0] == 0:
            #두번째 step일 때, 선택된 marine을 부대지정함
            #사실 현재 꼭필요는 없지만 이후 두 marine 모두 이용하는 version에서 필요함
            return actions.FUNCTIONS.select_control_group([CONTROL_GROUP_SET], [MARINE_GROUP_ORDER])
        #환경 전처리
        state = get_state(obs)
        screen = torch.tensor(state).unsqueeze(0).float().to(device)
        #action probability와 value 얻음
        action_prob,value,h_out = self.network(screen,h_in)
        action_dist = Categorical(action_prob)
        #action sampling
        action_coords = action_dist.sample().reshape(-1,1)
        #action -> x,y coordinates로 바꿈
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
        first_hidden  = (h1_in.detach(), h2_in.detach()) #(1,1,4096),(1,1,4096)
        second_hidden = (h1_out.detach(), h2_out.detach()) #(1,1,4096),(1,1,4096)
        
        for i in range(K_EPOCH):
            #1. self.network에서 나온 h_out이용한 inference
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
                h_out = (torch.zeros(1,1,4 * int(SCREEN_SIZE/2) * int(SCREEN_SIZE/2)).to(device),
                torch.zeros(1,1,4 * int(SCREEN_SIZE/2) * int(SCREEN_SIZE/2)).to(device))
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
