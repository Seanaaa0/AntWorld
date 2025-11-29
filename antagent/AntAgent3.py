import numpy as np
from collections import deque
import random


class AntAgent3:
    """
    螞蟻個體：
    - memory: 0 = 未知, 1 = 已知空地, 2 = 食物, 3 = 牆/巢
    - mode : "explore" / "return" / "done"
    """

    def __init__(self, agent_id, pos, is_explorer=True, size=150):
        self.id = agent_id
        self.pos = pos  # [x, y]
        self.size = size

        self.carrying_food = False
        self.is_explorer = is_explorer
        self.steps_taken = 0
        self.max_steps = 300
        self.mode = "explore"  # or "return" or "done"
        self.return_path = []  # planned path home

        # 0 = 未知, 1 = 已知空地, 2 = 食物, 3 = 巢/牆
        self.memory = np.zeros((self.size, self.size), dtype=np.int8)
        self.path_history = [tuple(pos)]
        self.blocked_count = 0

    # ------------------------------------------------------------------ #
    # 感知
    # ------------------------------------------------------------------ #
    def observe(self, global_grid):
        """
        從全域地圖更新自己附近 3x3 的記憶。

        映射規則：
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
                        self.memory[nx][ny] = max(self.memory[nx][ny], 3)
                    else:                 # 空地
                        self.memory[nx][ny] = max(self.memory[nx][ny], 1)

    # ------------------------------------------------------------------ #
    # 決策：加入「聞食物氣味」
    # ------------------------------------------------------------------ #
    def decide_move(self, scent_grid=None):
        """
        回巢模式：沿 return_path 走。
        探索模式：
        1) 若聞到食物氣味，優先往氣味最濃的方向走。
        2) 聞不到時，優先探索 memory == 0 的未知格。
        3) 再不然隨機亂走。
        """
        # 回巢中：沿著規劃好的路走
        if self.mode == "return":
            if not self.return_path:
                return (0, 0)
            target = self.return_path.pop(0)
            dx = target[0] - self.pos[0]
            dy = target[1] - self.pos[1]
            return (dx, dy)

        # 探索模式
        directions = [
            (dx, dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]
        random.shuffle(directions)

        x, y = self.pos

        # 1) 聞氣味：只在「沒搬食物」且提供 scent_grid 時使用
        if scent_grid is not None and not self.carrying_food:
            best_dirs = []
            best_scent = 0.0

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    s = scent_grid[nx][ny]
                    if s > best_scent:
                        best_scent = s
                        best_dirs = [(dx, dy)]
                    elif s == best_scent and s > 0:
                        best_dirs.append((dx, dy))

            if best_scent > 0 and best_dirs:
                return random.choice(best_dirs)

        # 2) 優先走向記憶中未知的格子
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.memory[nx][ny] == 0:
                    return (dx, dy)

        # 3) 真的沒地方好走就隨機
        return random.choice(directions)

    # ------------------------------------------------------------------ #
    # 回巢相關
    # ------------------------------------------------------------------ #
    def should_return(self):
        return self.mode == "explore" and self.steps_taken >= self.max_steps

    def reset_steps(self):
        self.steps_taken = 0

    def known_food_locations(self):
        return list(zip(*np.where(self.memory == 2)))

    def plan_return_path(self, nest_coords):
        """從自己的記憶中，用 BFS 找出回巢路徑，盡量不穿牆。"""
        queue = deque()
        visited = set()
        start = tuple(self.pos)
        queue.append((start, [start]))
        visited.add(start)

        def walkable(cell):
            x, y = cell
            if cell in nest_coords:
                return True
            # memory == 3 視為牆
            return self.memory[x][y] != 3

        while queue:
            current, path = queue.popleft()
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

        return False  # 找不到路

    # ------------------------------------------------------------------ #
    # 食物區塊標記
    # ------------------------------------------------------------------ #
    def mark_food_region(self, start_pos, global_grid):
        """從起始格出發，尋找並標記整塊食物區域。"""
        queue = deque([start_pos])
        visited = set()
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if 0 <= x < self.size and 0 <= y < self.size and global_grid[x][y] == 2:
                self.memory[x][y] = 2
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
