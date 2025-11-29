import numpy as np
from collections import deque
import random


class AntAgent:
    def __init__(self, agent_id, pos, is_explorer=True, size=150):
        self.id = agent_id
        self.pos = pos  # [x, y]
        self.size = size

        self.carrying_food = False
        self.is_explorer = is_explorer
        self.steps_taken = 0
        self.max_steps = 1000
        self.mode = "explore"  # or "return" or "done"
        self.return_path = []  # planned path home

        # 0 = 未知, 1 = 已知空地, 2 = 食物, 3 = 巢/牆
        self.memory = np.zeros((self.size, self.size), dtype=np.int8)
        self.path_history = [tuple(pos)]
        self.blocked_count = 0
        self.just_reset = False

    def observe(self, global_grid):
        """從全域地圖更新自己附近 3x3 的記憶。

        映射規則（你可以依你實際 grid 意義微調）：
        - global 0: 空地        -> memory 1 = 已知可走
        - global 1: 牆 / 巢穴   -> memory 3 = 不可走
        - global 2: 食物        -> memory 2 = 食物
        """
        x, y = self.pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    val = global_grid[nx][ny]
                    if val == 2:          # 食物
                        self.memory[nx][ny] = 2
                    elif val == 1:        # 牆或巢穴
                        # 用 max 避免把食物等資訊蓋掉
                        self.memory[nx][ny] = max(self.memory[nx][ny], 3)
                    else:                 # 空地
                        self.memory[nx][ny] = max(self.memory[nx][ny], 1)

    def decide_move(self):
        # 回巢模式：沿著 return_path 走
        if self.mode == "return":
            if not self.return_path:
                return (0, 0)  # 沒路就先原地
            target = self.return_path.pop(0)
            dx = target[0] - self.pos[0]
            dy = target[1] - self.pos[1]
            return (dx, dy)

        # explore 模式
        directions = [
            (dx, dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]
        random.shuffle(directions)

        # 優先走向「記憶中未知」的格子
        for dx, dy in directions:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.memory[nx][ny] == 0:
                    return (dx, dy)

        # 如果附近都看過了，就隨便走一個
        return random.choice(directions)

    def move(self, direction, global_grid, agent_positions):
        """目前沒有被 env_interface_2 用到，但保留以後擴充。"""
        new_x = self.pos[0] + direction[0]
        new_y = self.pos[1] + direction[1]

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if global_grid[new_x][new_y] != 1 and (new_x, new_y) not in agent_positions:
                self.pos = [new_x, new_y]
                self.steps_taken += 1
                self.path_history.append((new_x, new_y))
                self.blocked_count = 0
                return True
            else:
                self.blocked_count += 1
                print(
                    f"[{self.id}] 移動到 ({new_x},{new_y}) 被擋住（連續 {self.blocked_count} 次）"
                )
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
        """用自己的記憶規劃一條『不穿牆』的回巢路徑。

        nest_coords: 巢穴所有格子的座標 list[(x, y), ...]
        回傳 True/False 表示是否找得到路。
        """
        from collections import deque

        start = tuple(self.pos)
        queue = deque()
        visited = set()

        queue.append((start, [start]))
        visited.add(start)

        def walkable(cell):
            x, y = cell
            # 目標是巢穴：無論 memory 標什麼都允許走進去
            if cell in nest_coords:
                return True
            # memory == 3 我們當成牆 / 不可走
            return self.memory[x][y] != 3

        while queue:
            current, path = queue.popleft()

            # 抵達巢穴：把 path 存成 return_path（去掉當前位置）
            if current in nest_coords:
                self.return_path = path[1:]
                self.mode = "return"
                self.blocked_count = 0
                return True

            x, y = current
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        nxt = (nx, ny)
                        if nxt not in visited and walkable(nxt):
                            visited.add(nxt)
                            queue.append((nxt, path + [nxt]))

        # 找不到路：維持原本模式，回傳 False
        return False

    def mark_food_region(self, start_pos, global_grid):
        """從起始格出發，尋找並標記整塊食物區域。"""
        queue = deque([start_pos])
        visited = set()
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if (
                0 <= x < self.size
                and 0 <= y < self.size
                and global_grid[x][y] == 2
            ):
                self.memory[x][y] = 2
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
