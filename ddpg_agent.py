# ddpg_agent
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
np.random.seed(0)
# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# Actor network
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = T.tanh(self.mu(x))
        return mu


# Critic network
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)  # اصلاح شد
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)   # ابعاد [batch, state_dim+action_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q


# DDPG Agent
class DDPGAgent:
    def __init__(self, input_dims, num_actions, tau, gamma, max_size,
                 hidden1_dims, hidden2_dims, batch_size, critic_lr, actor_lr):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, num_actions)
        self.batch_size = batch_size
        self.n_actions = num_actions

        self.actor = ActorNetwork(actor_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.critic = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)

        self.target_actor = ActorNetwork(actor_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.target_critic = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)
        self.actor.train()
        noise = np.random.normal(scale=0.1, size=self.n_actions)  # نویز اکتشاف
        mu_prime = mu.cpu().detach().numpy()[0] + noise
        return np.clip(mu_prime, -1, 1)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, terminal = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        terminal = T.tensor(terminal, dtype=T.float).to(self.actor.device)

        target_actions = self.target_actor(states_)
        critic_value_ = self.target_critic(states_, target_actions)
        critic_value = self.critic(states, actions)

        target = rewards + self.gamma * critic_value_ * terminal
        critic_loss = F.mse_loss(target, critic_value)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        for name in actor_params:
            actor_params[name] = tau * actor_params[name].clone() + \
                                 (1 - tau) * target_actor_params[name].clone()
        for name in critic_params:
            critic_params[name] = tau * critic_params[name].clone() + \
                                  (1 - tau) * target_critic_params[name].clone()

        self.target_actor.load_state_dict(actor_params)
        self.target_critic.load_state_dict(critic_params)