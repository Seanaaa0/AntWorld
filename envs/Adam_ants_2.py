import numpy as np
import random


class AntWorldEnv:
    def __init__(self, size=150, nest_size=4, food_size=9, min_dist=10, seed=None):
        self.size = size
        self.nest_size = nest_size
        self.food_size = food_size
        self.min_dist = min_dist
        self.rng = random.Random(seed)
        # 0: 空地, 1: 蟻窩, 2: 食物
        self.grid = np.zeros((size, size), dtype=np.int8)

        self.nest_pos = self._place_nest()
        self.food_positions = []
        self._place_multiple_food_zones(num=4)

    def _place_nest(self):
        x = self.rng.randint(0, self.size - self.nest_size)
        y = self.rng.randint(0, self.size - self.nest_size)
        self.grid[x:x+self.nest_size, y:y+self.nest_size] = 1
        return (x, y)

    def _place_multiple_food_zones(self, num=4):
        attempts = 0
        while len(self.food_positions) < num and attempts < 100:
            fx = self.rng.randint(0, self.size - self.food_size)
            fy = self.rng.randint(0, self.size - self.food_size)
            center_food = (fx + self.food_size // 2, fy + self.food_size // 2)
            nx, ny = self.nest_pos
            center_nest = (nx + self.nest_size // 2, ny + self.nest_size // 2)
            dist = np.linalg.norm(
                np.array(center_food) - np.array(center_nest))
            if dist >= self.min_dist:
                if np.all(self.grid[fx:fx+self.food_size, fy:fy+self.food_size] == 0):
                    self.grid[fx:fx+self.food_size, fy:fy+self.food_size] = 2
                    self.food_positions.append((fx, fy))
            attempts += 1

    def get_grid(self):
        return self.grid.copy()

    def render_ascii(self):
        symbols = {0: ".", 1: "N", 2: "F"}
        for row in self.grid:
            print("".join([symbols[val] for val in row]))


if __name__ == "__main__":
    env = AntWorldEnv(seed=random)
    env.render_ascii()
