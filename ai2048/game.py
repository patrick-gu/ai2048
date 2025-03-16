import random


class Game:
    """2048 game state"""

    state: list[list[int]]

    def __init__(self):
        self.state = [[0] * 4 for _ in range(4)]
        self._place()
        self._place()

    def _place(self):
        places = []
        for i in range(4):
            for j in range(4):
                if self.state[i][j] == 0:
                    places.append((i, j))
        if len(places) == 0:
            return
        x, y = random.choice(places)
        num = random.choices([2, 4], [0.9, 0.1])[0]
        self.state[x][y] = num

    def _move_left(self) -> bool:
        valid = False
        for i in range(4):
            last = [*self.state[i]]
            self.state[i] = ([e for e in self.state[i] if e != 0] + [0] * 4)[:4]
            j = 0
            for j in range(3):
                if self.state[i][j] == self.state[i][j + 1]:
                    self.state[i][j] *= 2
                    self.state[i] = (
                        self.state[i][: j + 1] + self.state[i][j + 2 :] + [0]
                    )
            if self.state[i] != last:
                valid = True
        if valid:
            self._place()
            return True
        return False

    def _rotate_ccw(self):
        self.state = [list(row) for row in zip(*self.state[::-1])]

    def move(self, direction: int) -> bool:
        """
        Play a move in the game. Return whether the move is valid.

        Directions: 0 - Right, 1 - Up, 2 - Left, 3 - Down.
        """

        for _ in range(direction + 2):
            self._rotate_ccw()
        valid = self._move_left()
        for _ in range(6 - direction):
            self._rotate_ccw()
        return valid

    def clone(self) -> "Game":
        g = Game()
        g.state = [row[:] for row in self.state]
        return g

    def valid(self, direction: int) -> bool:
        return self.clone().move(direction)

    def alive(self) -> bool:
        return any(0 in row for row in self.state)

    def display(self):
        for row in self.state:
            print(row)
