import numpy as np
import random


class AntWorldEnv:
    def __init__(self, size=150, nest_size=4, food_size=10, min_dist=30, seed=None):
        self.size = size
        self.nest_size = nest_size
        self.food_size = food_size
        self.min_dist = min_dist
        self.rng = random.Random(seed)
        # 0: 空地, 1: 蟻窩, 2: 食物
        self.grid = np.zeros((size, size), dtype=np.int8)

        self.nest_pos = self._place_nest()
        self.food_pos = self._place_food()

    def _place_nest(self):
        x = self.rng.randint(0, self.size - self.nest_size)
        y = self.rng.randint(0, self.size - self.nest_size)
        self.grid[x:x+self.nest_size, y:y+self.nest_size] = 1
        return (x, y)

    def _place_food(self):
        while True:
            fx = self.rng.randint(0, self.size - self.food_size)
            fy = self.rng.randint(0, self.size - self.food_size)
            # 確保距離大於 min_dist
            nx, ny = self.nest_pos
            center_food = (fx + self.food_size // 2, fy + self.food_size // 2)
            center_nest = (nx + self.nest_size // 2, ny + self.nest_size // 2)
            dist = np.linalg.norm(
                np.array(center_food) - np.array(center_nest))
            if dist >= self.min_dist:
                break
        self.grid[fx:fx+self.food_size, fy:fy+self.food_size] = 2
        return (fx, fy)

    def get_grid(self):
        return self.grid.copy()

    def render_ascii(self):
        symbols = {0: ".", 1: "N", 2: "F"}
        for row in self.grid:
            print("".join([symbols[val] for val in row]))


if __name__ == "__main__":
    from env_interface import AntSimInterface
    import time

    # 初始化模擬環境
    sim = AntSimInterface(seed=42)

    # 進行 100 步模擬
    for step in range(100):
        sim.step()

        # 若需要可視化，可取得當前狀態
        grid, ant_layer = sim.get_state()

        # 可選擇加入延遲
        time.sleep(0.1)
