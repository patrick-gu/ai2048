from ai2048.game import Game


if __name__ == "__main__":
    game = Game()

    key_mapping = {
        "w": 1,  # Up
        "d": 0,  # Right
        "s": 3,  # Down
        "a": 2,  # Left
    }

    game.display()

    while True:
        key = input()
        if key in key_mapping:
            direction = key_mapping[key]
            game.move(direction)
            game.display()
        else:
            break
