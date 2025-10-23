#models
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# from ipdb import set_trace as debug


def fanin_init(size, fanin=None):

    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return(torch.Tensor(size).uniform_(-v, v))


class Actor(nn.Module):
    def __init__(self, input_dims, num_actions, hidden1_dims, hidden2_dims, init_w=3e-3, chkpt_dir='./models', name="no_name_actor"):

        super(Actor, self).__init__()

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'actor_ddpg')

        self.fc1 = nn.Linear(input_dims[0], hidden1_dims)
        self.bn1 = nn.LayerNorm(hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.bn2 = nn.LayerNorm(hidden2_dims)

        self.pi = nn.Linear(hidden2_dims, num_actions)

        self.relu = nn.ReLU()

        self.init_weights(init_w)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def init_weights(self, init_w):

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.pi.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):

        out = self.fc1(state)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.pi(out)
        action = F.tanh(out)
        return(action)

    def save_checkpoint(self):
        print('<< saving actor... >>')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('<< loading actor... >>')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):

    def __init__(self, input_dims, num_actions, hidden1_dims=400, hidden2_dims=300, init_w=3e-3, chkpt_dir='./models', name="no_name_critic"):

        super(Critic, self).__init__()

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'critic_ddpg')

        self.fc1 = nn.Linear(input_dims[0], hidden1_dims)
        self.bn1 = nn.LayerNorm(hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims + num_actions, hidden2_dims)
        self.bn2 = nn.LayerNorm(hidden2_dims)

        self.Q = nn.Linear(hidden2_dims, 1)

        self.relu = nn.ReLU()

        self.init_weights(init_w)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def init_weights(self, init_w):

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.Q.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):

        out = self.fc1(state)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(torch.cat([out, action], 1))
        out = self.bn2(out)
        out = self.relu(out)

        q = self.Q(out)

        return(q)

    def save_checkpoint(self):
        print('<< saving critic... >>')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('<< loading critic... >>')
        self.load_state_dict(torch.load(self.checkpoint_file))