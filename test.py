import torch
import torch.nn as nn
import copy
import random
from collections import deque
from tqdm import tqdm
import numpy as np

class CustomNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(CustomNN, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQN_Agent:
    def __init__(self, seed, input_size, output_size, hidden_layers, lr, sync_freq, exp_replay_size, device):
        torch.manual_seed(seed)
        self.device = device
        self.q_net = CustomNN(input_size, output_size, hidden_layers).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().to(device)
        self.experience_replay = deque(maxlen=exp_replay_size)
        return

    def get_action(self, state, action_space_len, epsilon):
        obs = state[0] if isinstance(state, tuple) else state
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(obs).float().to(self.device))
        Q, A = torch.max(Qp, axis=0)
        A = A if torch.rand(1).item() > epsilon else torch.randint(0, action_space_len, (1,), device=self.device)
        return A

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state.to(self.device))
        q, _ = torch.max(qp, axis=1)
        return q

    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)

        sample = random.sample(self.experience_replay, sample_size)

        s_list, a_list, rn_list, sn_list = zip(*sample)

        s = torch.tensor(np.array([np.array(s[0]) if isinstance(s, tuple) else np.array(s) for s in s_list]),
                         dtype=torch.float32, device=self.device)
        a = torch.tensor([a for a in a_list], dtype=torch.float32, device=self.device)
        rn = torch.tensor([rn for rn in rn_list], dtype=torch.float32, device=self.device)

        sn = torch.tensor(np.array([list(sn.values()) if isinstance(sn, dict) else sn for sn in sn_list]),
                          dtype=torch.float32, device=self.device)

        return s, a, rn, sn

    def train(self, batch_size):
        state, action, reward, next_action = self.sample_from_experience(sample_size=batch_size)
        print(f"{reward[0] = } || {action[0] = } || {state[0] =  } || {next_action[0] = }")
        #s,     a,       rn,       sn = self.sample_from_experience(sample_size=batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        qp = self.q_net(state)
        pred_return, _ = torch.max(qp, axis=1)

        q_next = self.get_q_next(next_action)
        target_return = reward + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None

    def is_valid_move(self, action):
        return self.board[action] == ' ' and not self.winner

    def make_move(self, action):
        if self.is_valid_move(action):
            self.board[action] = self.current_player
            self.check_winner()
            self.switch_player()
            return True
        return False

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):

        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                                (0, 4, 8), (2, 4, 6)]

        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                self.winner = self.board[combo[0]]

                return self.winner

        if ' ' not in self.board:
            self.winner = 'Tie'
            return 'Tie'

        return None

    def get_state(self):
        return np.array([1 if c == 'X' else -1 if c == 'O' else 0 for c in self.board])

    def display_board(self):
        print(
            f"{self.board[0]} | {self.board[1]} | {self.board[2]}\n"
            f"---------\n"
            f"{self.board[3]} | {self.board[4]} | {self.board[5]}\n"
            f"---------\n"
            f"{self.board[6]} | {self.board[7]} | {self.board[8]}"
            f"\n\n")

# Main execution for Tic Tac Toe
env = TicTacToe()
input_dim = 9  # Tic Tac Toe board size
output_dim = 9  # Number of possible actions (each cell on the board)
exp_replay_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQN_Agent(seed=1423, input_size=input_dim, output_size=output_dim, hidden_layers=[64], lr=0.1, sync_freq=5,
                  exp_replay_size=exp_replay_size, device=device)

#index = 0
#for i in range(exp_replay_size):
#    state = env.get_state()
#    done = False
#    while not done:
#        A = agent.get_action(state, 9, epsilon=1)
#        action = A.item()
#        step_result = env.make_move(action)
#        obs_next = env.get_state()
#        reward = 1 if env.winner == 'X' else 0 if env.winner == 'Tie' else -1
#        agent.collect_experience([state, action, reward, obs_next])
#        state = obs_next
#        index += 1
#        if index > exp_replay_size:
#            break

losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
index = 128
episodes = 10000
epsilon = 1

#for i in tqdm(range(episodes)):
for i in range(episodes):

    state, done, losses, ep_len, reward_sum = env.get_state(), False, 0, 0, 0
    env.reset()
    while not done:
        state = env.get_state()
        ep_len += 1
        A = agent.get_action(state, 9, epsilon)
        action = A.item()
        valued_move = env.make_move(action)
        obs_next = env.get_state()
        if env.check_winner() != None:
            #print(not env.check_winner())
            env.reset()
        if np.count_nonzero(obs_next) == 9:
            done = True
        reward = 0
        # if env.winner == "X":  #
        #    reward = 1
        # elif env.winner == "O":  # if ki loss
        #    reward = -1
        reward += int((int(state[action] == 0) - 0.5) * 2)

        agent.collect_experience([state, action, reward, obs_next])

        reward_sum += reward
        index += 1
        if index > 128:
            index = 0
            for j in range(4):
                loss = agent.train(batch_size=128)
                losses += loss
        if epsilon < .2 or ep_len%1000 == 0:
            #print(reward_sum, epsilon)
            print(f"{ep_len = } || {epsilon = } || {reward_sum = } || {env.check_winner()} || {action = } || {state =  } || {obs_next = }")
            reward_sum = 0
    #print(f"{ep_len = } || {epsilon = } || {reward = } || {int(state[action] == 0)} || {action = } || {state =  } || {obs_next = }")

    if epsilon > 0.05 and ep_len % 100 == 0:
        epsilon *= 0.9

    losses_list.append(losses / ep_len), reward_list.append(reward_sum), episode_len_list.append(ep_len), epsilon_list.append(
        epsilon)

for loss_ in losses_list:
    print(loss_)
print(len(losses_list))
