import numpy as np
import random


class AntAgent:
    def __init__(self, agent_id, pos, is_explorer=True):
        self.id = agent_id
        self.pos = pos  # [x, y]
        self.carrying_food = False
        self.is_explorer = is_explorer
        self.steps_taken = 0
        self.max_steps = 300

        # 個人記憶（使用 0: 未知, 1: 可走, 2: 食物）
        self.memory = np.zeros((150, 150), dtype=np.int8)
        self.path_history = [tuple(pos)]  # 紀錄曾走過的點

    def observe(self, global_grid):
        x, y = self.pos
        self.memory[x][y] = global_grid[x][y] if 0 <= x < 150 and 0 <= y < 150 else 0

    def move(self, direction, global_grid):
        """ direction: tuple(dx, dy) """
        new_x = self.pos[0] + direction[0]
        new_y = self.pos[1] + direction[1]
        if 0 <= new_x < 150 and 0 <= new_y < 150 and global_grid[new_x][new_y] != 1:
            self.pos = [new_x, new_y]
            self.steps_taken += 1
            self.path_history.append((new_x, new_y))
            return True
        return False

    def should_return(self):
        return self.steps_taken >= self.max_steps

    def reset_steps(self):
        self.steps_taken = 0

    def known_food_locations(self):
        """從記憶中找出食物格子"""
        return list(zip(*np.where(self.memory == 2)))

    def decide_move(self):
        """優先探索未知格子，否則亂走"""
        directions = [(dx, dy) for dx in [-1, 0, 1]
                      for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        unexplored = []

        for dx, dy in directions:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < 150 and 0 <= ny < 150 and self.memory[nx][ny] == 0:
                unexplored.append((dx, dy))

        if unexplored:
            return random.choice(unexplored)
        else:
            return random.choice(directions)
