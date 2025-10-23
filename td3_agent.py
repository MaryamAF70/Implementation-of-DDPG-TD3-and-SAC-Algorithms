# td3_agent.py
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
np.random.seed(0)

# --- ReplayBuffer (همان کد) ---
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

# --- شبکه‌های مشترک (همان طرح کلی شما) ---
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

# --- TD3 Agent ---
class TD3Agent:
    def __init__(self, input_dims, num_actions, tau, gamma, max_size,
                 hidden1_dims, hidden2_dims, batch_size, critic_lr, actor_lr,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        # پارامترها مشابه DDPG با افزودنی‌های TD3 به صورت اختیاری
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, num_actions)
        self.batch_size = batch_size
        self.n_actions = num_actions

        # actor و دو critic
        self.actor = ActorNetwork(actor_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.critic1 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.critic2 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)

        # target networks
        self.target_actor = ActorNetwork(actor_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.target_critic1 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)
        self.target_critic2 = CriticNetwork(critic_lr, input_dims, hidden1_dims, hidden2_dims, num_actions)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.learn_step_counter = 0

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)
        self.actor.train()
        noise = np.random.normal(scale=0.1, size=self.n_actions)
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
        rewards = T.tensor(rewards, dtype=T.float).unsqueeze(1).to(self.actor.device)  # [batch,1]
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        terminal = T.tensor(terminal, dtype=T.float).unsqueeze(1).to(self.actor.device)

        # target actions with clipped noise
        with T.no_grad():
            noise = (T.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(states_) + noise).clamp(-1, 1)

            target_q1 = self.target_critic1(states_, next_actions)
            target_q2 = self.target_critic2(states_, next_actions)
            target_q = T.min(target_q1, target_q2)
            target = rewards + self.gamma * target_q * terminal  # shapes match [batch,1]

        # current Q estimates
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # critic losses
        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)

        self.critic1.optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2.optimizer.step()

        # delayed policy updates
        if self.learn_step_counter % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # soft update targets
            self.update_network_parameters()

        self.learn_step_counter += 1

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # actor params
        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        for name in actor_params:
            target_actor_params[name] = tau * actor_params[name].clone() + (1 - tau) * target_actor_params[name].clone()
        self.target_actor.load_state_dict(target_actor_params)

        # critic1 params
        critic1_params = dict(self.critic1.named_parameters())
        target_critic1_params = dict(self.target_critic1.named_parameters())
        for name in critic1_params:
            target_critic1_params[name] = tau * critic1_params[name].clone() + (1 - tau) * target_critic1_params[name].clone()
        self.target_critic1.load_state_dict(target_critic1_params)

        # critic2 params
        critic2_params = dict(self.critic2.named_parameters())
        target_critic2_params = dict(self.target_critic2.named_parameters())
        for name in critic2_params:
            target_critic2_params[name] = tau * critic2_params[name].clone() + (1 - tau) * target_critic2_params[name].clone()
        self.target_critic2.load_state_dict(target_critic2_params)