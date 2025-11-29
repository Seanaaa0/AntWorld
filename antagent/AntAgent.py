import numpy as np
from collections import deque
import random


class AntAgent:
    def __init__(self, agent_id, pos, is_explorer=True):
        self.id = agent_id
        self.pos = pos  # [x, y]
        self.carrying_food = False
        self.is_explorer = is_explorer
        self.steps_taken = 0
        self.max_steps = 300
        self.mode = "explore"  # or "return"
        self.return_path = []  # planned path home

        self.memory = np.zeros((150, 150), dtype=np.int8)
        self.path_history = [tuple(pos)]
        self.blocked_count = 0
        self.just_reset = False

    def observe(self, global_grid):
        x, y = self.pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 150 and 0 <= ny < 150:
                    self.memory[nx][ny] = global_grid[nx][ny]

    def decide_move(self):
        if self.mode == "return":
            if not self.return_path:
                return (0, 0)  # 被卡住但仍在 return 模式
            target = self.return_path.pop(0)
            dx = target[0] - self.pos[0]
            dy = target[1] - self.pos[1]
            return (dx, dy)

        # explore 模式
        directions = [(dx, dy) for dx in [-1, 0, 1]
                      for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < 150 and 0 <= ny < 150 and self.memory[nx][ny] == 0:
                return (dx, dy)

        return random.choice(directions)

    def move(self, direction, global_grid, agent_positions):
        new_x = self.pos[0] + direction[0]
        new_y = self.pos[1] + direction[1]

        if 0 <= new_x < 150 and 0 <= new_y < 150:
            if global_grid[new_x][new_y] != 1 and (new_x, new_y) not in agent_positions:
                self.pos = [new_x, new_y]
                self.steps_taken += 1
                self.path_history.append((new_x, new_y))
                self.blocked_count = 0
                return True
            else:
                self.blocked_count += 1
                print(
                    f"[{self.id}] 移動到 ({new_x},{new_y}) 被擋住（連續 {self.blocked_count} 次）")
                if self.blocked_count >= 3:
                    print(f"[{self.id}] 被卡 {self.blocked_count} 次，強制切回探索")
                    self.return_path = []
                    self.mode = "explore"
                    self.reset_steps()
        return False

    def should_return(self):
        return self.mode == "explore" and self.steps_taken >= self.max_steps

    def reset_steps(self):
        self.steps_taken = 0

    def known_food_locations(self):
        return list(zip(*np.where(self.memory == 2)))

    def plan_return_path(self, nest_coords):
        """從自己的記憶中，用 BFS 找出回巢路徑"""
        queue = deque()
        visited = set()
        queue.append((tuple(self.pos), [tuple(self.pos)]))
        visited.add(tuple(self.pos))

        while queue:
            current, path = queue.popleft()
            if current in nest_coords:
                self.return_path = path[1:]  # 排除自己
                self.mode = "return"
                return True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == dy == 0:
                        continue
                    nx, ny = current[0] + dx, current[1] + dy
                    if 0 <= nx < 150 and 0 <= ny < 150:
                        if (nx, ny) not in visited:
                            queue.append(((nx, ny), path + [(nx, ny)]))
                            visited.add((nx, ny))

        return False  # 找不到

    def mark_food_region(self, start_pos, global_grid):
        """從起始格出發，尋找並標記整塊食物區域"""
        queue = deque([start_pos])
        visited = set()
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if 0 <= x < 150 and 0 <= y < 150 and global_grid[x][y] == 2:
                self.memory[x][y] = 2
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
