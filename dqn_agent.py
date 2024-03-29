import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from q_network import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, episodes=10_000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 8)  # Random action
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return int(torch.argmax(q_values).item())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(np.array(batch[0]), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device)
        next_state_batch = torch.tensor(np.array(batch[2]), device=self.device, dtype=torch.float32)
        reward_batch = torch.tensor(batch[3], device=self.device, dtype=torch.float32)

        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)