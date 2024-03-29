# downloader.py
import time
import matplotlib.pyplot as plt
import torch
from dqn_agent import DQNAgent
from tictactoe import TicTacToe
from tqdm import tqdm

def train_agent(num_episodes, plot=False):
    env = TicTacToe()
    batch_size = 64
    win_rates = []
    rewards = []

    for episode in tqdm(range(num_episodes)):
        env.reset()
        state = env.get_state()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            new_state, reward, terminated, truncated = env.step(action)
            agent.replay_buffer.push(state, action, new_state, reward)
            total_reward += reward
            state = new_state
            rewards.append(reward)
            done = terminated or truncated

        agent.train()

        if episode % 1000 == 0:
            win_rate = env.wins / (episode + 1)
            win_rates.append(win_rate)
            print(f"Episode: {episode}, Win Rate: {win_rate:.2f}, Reward Ø: {sum(rewards[-1000:])}")

    if plot:
        plot_rewards(rewards)
        plot_win_rates(win_rates)

    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'model.pth')

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def plot_win_rates(win_rates):
    plt.figure(figsize=(10, 5))
    plt.plot(win_rates)
    plt.title('Win Rate over Episodes')
    plt.xlabel('Episode (x100)')
    plt.ylabel('Win Rate')
    plt.show()

def play_vs_ai(agent):
    env = TicTacToe()
    state = env.get_state()
    env.display_board()

    while not env.winner:
        if env.current_player == 'O':
            action = int(input("Your move (0-8): "))
            if not env.is_valid_move(action):
                print("Invalid move. Please try again.")
                continue
        else:
            action = agent.select_action(state)
            print(f"AI's move: {action}")

        _, reward, _, _ = env.step(action, play=True)
        print(reward)
        state = env.get_state()
        env.display_board()

    if env.winner == 'Tie':
        print("It's a tie!")
    else:
        print(f"{env.winner} wins!")

if __name__ == "__main__":
    num_episodes = 1_000_000
    agent = DQNAgent(num_episodes)

    start_time = time.perf_counter()
    train_agent(num_episodes, plot=True)
    end_time = time.perf_counter()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    while True:
        play_vs_ai(agent)