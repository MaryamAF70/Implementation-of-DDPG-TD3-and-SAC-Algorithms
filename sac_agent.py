# sac_agent.py
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
np.random.seed(0)

# --- ReplayBuffer همانند بالا ---
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

# --- CriticNetwork همان طرح ---
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

# --- SAC actor: stochastic (mu, log_std) ---
class SACActor(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, max_action=1.0, log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = T.exp(log_std)
        # reparameterization
        eps = T.randn_like(mu)
        action = mu + eps * std
        # squashed action
        tanh_action = T.tanh(action)
        log_prob = -0.5 * ((eps ** 2) + 2 * log_std + T.log(2 * T.tensor(np.pi)))
        log_prob = log_prob.sum(dim=1, keepdim=True)
        # correction for tanh
        log_prob = log_prob - T.log(1 - tanh_action.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        return tanh_action * self.max_action, log_prob

# --- SAC Agent ---
class SACAgent:
    def __init__(self, input_dims, num_actions, tau, gamma, max_size,
                 hidden1_dims, hidden2_dims, batch_size, critic_lr, actor_lr,
                 alpha=0.2, automatic_entropy_tuning=False):
        # سازنده با ترتیب مشابه DDPG؛ alpha پیش‌فرض دارد تا اگر در common_params نبود خطا ندهد
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, num_actions)
        self.batch_size = batch_size
        self.n_actions = num_actions

        # networks: actor stochastic + two critics
        self.actor = SACActor(actor_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.critic1 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.critic2 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)

        self.target_critic1 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.target_critic2 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)

        # entropy temp
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            # target entropy = -dim(A)
            self.target_entropy = -num_actions
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=actor_lr)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        if evaluate:
            mu, _ = self.actor.forward(state)
            a = T.tanh(mu).cpu().detach().numpy()[0]
            return np.clip(a, -1, 1)
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, terminal = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).unsqueeze(1).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        terminal = T.tensor(terminal, dtype=T.float).unsqueeze(1).to(self.actor.device)

        # sample next actions from actor (stochastic) for target
        with T.no_grad():
            next_actions, next_log_pi = self.actor.sample(states_)
            target_q1 = self.target_critic1(states_, next_actions)
            target_q2 = self.target_critic2(states_, next_actions)
            target_q = T.min(target_q1, target_q2) - self.alpha * next_log_pi
            target = rewards + self.gamma * target_q * terminal

        # current Q estimates and critics update
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)

        self.critic1.optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2.optimizer.step()

        # actor update
        new_actions, log_pi = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = T.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_pi - q_new).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # optional automatic alpha tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # soft update of targets
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # critic1
        critic1_params = dict(self.critic1.named_parameters())
        target_critic1_params = dict(self.target_critic1.named_parameters())
        for name in critic1_params:
            target_critic1_params[name] = tau * critic1_params[name].clone() + (1 - tau) * target_critic1_params[name].clone()
        self.target_critic1.load_state_dict(target_critic1_params)

        # critic2
        critic2_params = dict(self.critic2.named_parameters())
        target_critic2_params = dict(self.target_critic2.named_parameters())
        for name in critic2_params:
            target_critic2_params[name] = tau * critic2_params[name].clone() + (1 - tau) * target_critic2_params[name].clone()
        self.target_critic2.load_state_dict(target_critic2_params)