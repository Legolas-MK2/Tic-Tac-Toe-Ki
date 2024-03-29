import random

import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None
        self.terminated = False
        self.truncated = False
        self.moves = 0
        self.max_moves = 100
        self.wins = 0

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None
        self.terminated = False
        self.truncated = False
        self.moves = 0

    def is_valid_move(self, action):
        return self.board[action] == ' ' and not self.winner

    def check_next_move_winner(self):

        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                                (0, 4, 8), (2, 4, 6)]

        player_symbol = "X" if self.current_player == "O" else "O"

        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == player_symbol and self.board[combo[2]] == ' ':
                return True
            elif self.board[combo[0]] == self.board[combo[2]] == player_symbol and self.board[combo[1]] == ' ':
                return True
            elif self.board[combo[1]] == self.board[combo[2]] == player_symbol and self.board[combo[0]] == ' ':
                return True

        return False

    def step(self, action, play=False):
        self.moves += 1
        reward = 0

        if not self.is_valid_move(action):
            reward = -100  # Large negative reward for invalid moves
        else:
            self.board[action] = self.current_player

            if self.check_winner() == self.current_player:
                reward = 100  # Large positive reward for winning
                self.terminated = True
            elif ' ' not in self.board:
                reward = 10  # Small positive reward for a draw
                self.terminated = True
            else:
                if self.check_next_move_winner():
                    reward = -50  # Medium negative reward for allowing the opponent to win
                else:
                    reward = 1  # Small positive reward for a valid move
                #if not play:
                #    self.board[random.sample([i for i, x in enumerate(self.board) if x == " "], 1)[0]] = "O"
                #else:
                self.switch_player()

        if self.moves > 20:
            self.truncated = True

        return self.get_state(), reward, self.terminated, self.truncated

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                                (0, 4, 8), (2, 4, 6)]

        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                self.winner = self.board[combo[0]]
                self.wins += 1
                return self.winner

        if ' ' not in self.board:
            self.winner = 'Tie'
            return 'Tie'

        return None

    def get_state(self):
        return np.array([1 if c == 'X' else -1 if c == 'O' else 0 for c in self.board])

    def display_board(self):
        tmp_board = [""]*9
        for indx, feld in enumerate(self.board):
            tmp_board[indx] = indx if feld == " " else feld

        print(
            f"{tmp_board[0]} | {tmp_board[1]} | {tmp_board[2]}\n"
            f"---------\n"
            f"{tmp_board[3]} | {tmp_board[4]} | {tmp_board[5]}\n"
            f"---------\n"
            f"{tmp_board[6]} | {tmp_board[7]} | {tmp_board[8]}"
            f"\n\n")

