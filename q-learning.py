import os.path
import pickle
import random
import time
import tkinter as tk
from tkinter import messagebox

import numpy
from tqdm import tqdm

from tictactoe import TicTacToe
import numpy as np


class QL:
    def __init__(self, epoch=0, table=np.zeros((19_683, 9)),random_moves=False):
        self.table = table
        self.lr = .1
        self.gamma = .4
        self.epsilon = .9
        self.epsilon_decay_rate = self.epsilon / epoch / 1.5
        self.random_moves=random_moves

    def get_action(self, state, print_: bool = False):
        if self.epsilon > random.random() or self.random_moves:
            return random.randint(0, 8)
        else:
            if print_:
                print(self.table[convert_to_number(state), :])
            return np.argmax(self.table[convert_to_number(state), :])

    def evaluate(self, action, state, new_state, reward):
        self.table[convert_to_number(state), action] = self.table[convert_to_number(state), action] + self.lr * (
                reward + self.gamma * np.max(self.table[convert_to_number(new_state), :]) -
                self.table[convert_to_number(state), action]
        )

    def step(self, i):
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0.001)


def convert_to_number(lst):
    """
    Convert a nine-element list where each element can be -1, 0, or 1
    into a number.
    """
    if len(lst) != 9:
        raise ValueError("Input list must have exactly nine elements.")

    num = 0
    for i, digit in enumerate(lst):
        if digit not in [-1, 0, 1]:
            raise ValueError("Each element in the list must be -1, 0, or 1.")
        num += (digit + 1) * (3 ** i)
    return num


def ki_vs_ki_train(table=None):
    reward_sum = 0
    if type(table) == type(numpy.zeros(1)):
        ki_x = QL(episodes, table)
        ki_o = QL(episodes, table, random_moves=True)
    else:
        ki_x = QL(episodes)
        ki_o = QL(episodes)

    for i in tqdm(range(episodes)):

        env.reset()
        state = env.get_state()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if env.current_player == "X":
                action = ki_x.get_action(state)
                new_state, reward, terminated, truncated = env.step(action)
                ki_x.evaluate(action, state, new_state, reward)
                #ki_o.evaluate(action, state, new_state, reward * -1 if reward > 2 else 0)
                reward_sum += reward
            else:
                action = ki_o.get_action(state)
                new_state, reward, terminated, truncated = env.step(action)
                ki_o.evaluate(action, state, new_state, reward)
                #ki_x.evaluate(action, state, new_state, reward * -1 if reward > 2 else 0)

            if i > episodes:
                for row in [0, 3, 6]:
                    print(f"{state[row]}|{state[row+1]}|{state[row+2]}          "
                          f"{new_state[row]}|{new_state[row+1]}|{new_state[row+2]}")
                    print("-"*5, " "*8, "-"*5)
                print(f"{action = } || {reward}")
                print("\n")
            state = new_state
        ki_x.step(i)
        ki_o.step(i)
        #if i % 10000 == 0:
        #    print(f"{reward_sum = } || {ki_x.epsilon} || {ki_o.epsilon}")
        #    reward_sum = 0
    return ki_x, ki_o


def play(ki: QL):
    env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        if env.current_player == 'O':
            env.display_board()
            action = int(input("Ihr Zug (0-8): "))
        else:
            action = ki.get_action(env.get_state(), True)
        new_state, reward, terminated, truncated = env.step(action)
        print(f"Der Player {env.current_player} hat ein reward von {reward}")
    env.display_board()


def main():
    force_train = True

    if force_train and os.path.exists("q-table_x.pkl"):
        os.remove("q-table_x.pkl")
        os.remove("q-table_o.pkl")

    if os.path.exists("q-table_x.pkl"):
        with open("q-table_x.pkl", "rb") as f:
            ki_x = pickle.load(f)

    if os.path.exists("q-table_o.pkl"):
        with open("q-table_o.pkl", "rb") as f:
            ki_o = pickle.load(f)

    ki_x, ki_o = ki_vs_ki_train()
    ki = ki_x
    #with open("q-table_x.pkl", "wb") as f:
    #    pickle.dump(ki_x, f)
    #with open("q-table_o.pkl", "wb") as f:
    #    pickle.dump(ki_o, f)
    while True:
        try:
            play(ki)
            if input("'s' for switch: ") == "s":
                if ki == ki_o:
                    ki = ki_x
                    print("ki ist 'x'")
                else:
                    ki = ki_o
                    print("ki ist 'o'")
        except Exception as e:
            print(e)


env = TicTacToe()
episodes = 5_000_000


if __name__ == "__main__":
    main()
