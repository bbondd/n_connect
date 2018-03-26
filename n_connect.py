import tensorflow as tf
import numpy as np


class Constant(object):
    class Board(object):
        ROW_SIZE = 6
        COL_SIZE = 7

        WIN_NUMBER = 4

    class Model(object):
        class Dense(object):
            LAYER_SIZE = 16
            UNIT_SIZE = 128

        class Dropout(object):
            RATE = 0.5

    class Player(object):
        A = 0
        B = 1
        DRAW = 'draw'


class Game(object):
    class Players(object):
        class Player(object):
            def __init__(self, value):
                self.value = value
                self.board_log = list()
                self.action_log = list()
                self.model_input_log = list()

        def __init__(self):
            self.A = self.Player(Constant.Player.A)
            self.B = self.Player(Constant.Player.B)
            self.A.next_player = self.B
            self.B.next_player = self.A

            self.Draw = self.Player(Constant.Player.DRAW)

    def __init__(self):
        self.players = self.Players()
        self.current_player = self.players.A
        self.current_board = {self.players.A: np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE]),
                              self.players.B: np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE])}

    def choice_to_location(self, choice):
        self.get_available_location(self.current_board)


    def put_stone(self, choice):
        self.current_player.board_log.append({
                self.players.A: self.current_board[self.players.A].copy(),
                self.players.B: self.current_board[self.players.B].copy(),
            }) #deep copy
        self.current_player.action.append(choice)


        self.current_board[self.current_player][location] = True

        def is_current_player_winner():
            for i in range(Constant.Board.WIN_NUMBER):
                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0]]
                        [location[1] - i + j]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[1] - i >= 0:
                        return True

                except IndexError:
                    pass

                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0] - i + j]
                        [location[1]]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[0] - i >= 0:
                        return True

                except IndexError:
                    pass

                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0] - i + j]
                        [location[1] - i + j]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[0] - i >= 0 and location[1] - i >= 0:
                        return True

                except IndexError:
                    pass

                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0] + i - j]
                        [location[1] - i + j]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[0] + i - Constant.Board.WIN_NUMBER + 1 >= 0\
                            and location[1] - i >= 0:
                        return True

                except IndexError:
                    pass

            return False

        if is_current_player_winner():
            return self.current_player

        else:
            if not self.get_available_location(self.current_board).any():
                return self.players.Draw
            else:
                self.current_player = self.current_player.next_player
                return None

    def get_available_location(self, board):
        return np.where((board[self.players.A] + board[self.players.B]) == True, False, True)

    def put_stone_by_model(self, model, p):
        model_input = list()
        model_input.append(self.current_board[self.players.A])
        model_input.append(self.current_board[self.players.B])
        for i in range(Constant.Model.input_turn_number - 1):
            try:
                model_input.append(self.players.A.board_log[-i - 1][self.players.A])
            except IndexError:
                model_input.append(np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE]))

            try:
                model_input.append(self.players.B.board_log[-i - 1][self.players.B])
            except IndexError:
                model_input.append(np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE]))
        model_input.append(np.full([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE], self.current_player.value))

        model_input = np.swapaxes(np.swapaxes(model_input, 0, 1), 1, 2)

        self.current_player.model_input_log.append(model_input)

        if np.random.rand() < p:
            prediction = np.multiply(
                np.random.rand(Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE),
                self.get_available_location(self.current_board)
            )
        else:
            prediction = np.multiply(
                model.predict_on_batch(np.array([model_input]))[0],
                self.get_available_location(self.current_board),
            )
        location = np.unravel_index(prediction.argmax(), prediction.shape)
        return self.put_stone(location)

    def print_board(self):
        print(self.current_board[self.players.A])
        print(self.current_board[self.players.B])

