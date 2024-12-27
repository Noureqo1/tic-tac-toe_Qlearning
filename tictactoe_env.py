import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1  # 1 for X, -1 for O
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return str(self.board.flatten().tolist())

    def get_valid_actions(self):
        if self.done:
            return []
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def make_move(self, action):
        if self.done:
            return self.get_state(), 0, True

        i, j = action
        if self.board[i][j] != 0:
            return self.get_state(), -10, True  # Invalid move penalty

        self.board[i][j] = self.current_player
        
        # Check for win
        if self._check_win():
            self.done = True
            self.winner = self.current_player
            return self.get_state(), 1, True  # Reward for winning

        # Check for draw
        if len(self.get_valid_actions()) == 0:
            self.done = True
            return self.get_state(), 0.5, True  # Small reward for draw

        self.current_player *= -1
        return self.get_state(), 0, False

    def _check_win(self):
        # Check rows, columns and diagonals
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True
        if abs(sum(np.diag(self.board))) == 3 or abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return True
        return False

    def render(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print('-' * 13)
        for i in range(3):
            row = '| '
            for j in range(3):
                row += f'{symbols[self.board[i][j]]} | '
            print(row)
            print('-' * 13)
