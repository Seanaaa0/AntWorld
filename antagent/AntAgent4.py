import numpy as np
from collections import deque
import random


class AntAgent4:
    """
    螞蟻個體：
    - memory: 0 = 未知, 1 = 已知空地, 2 = 食物, 3 = 牆/巢
    - mode  : "explore" / "return" / "return_to_food" / "done"
    """

    def __init__(self, agent_id, pos, is_explorer=True, size=150):
        self.id = agent_id
        self.pos = pos  # [x, y]
        self.size = size

        self.carrying_food = False
        self.is_explorer = is_explorer

        self.steps_taken = 0
        self.max_steps = 300  # 走太久就會想回家
        self.mode = "explore"  # "explore" / "return" / "return_to_food" / "done"
        self.return_path = []  # 規劃好的路徑（回巢或回食物點皆共用）

        # 0 = 未知, 1 = 已知空地, 2 = 食物, 3 = 巢/牆
        self.memory = np.zeros((self.size, self.size), dtype=np.int8)
        self.path_history = [tuple(pos)]
        self.blocked_count = 0

        # 最近一次找到的食物產地（給「回到原食物點」使用）
        self.last_food_cluster = None        # (x, y)
        self.last_food_timestamp = -1        # 由外部環境設定（env tick）

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
    # 決策（含：回巢 / 回食物點 / 嗅覺導引）
    # ------------------------------------------------------------------ #
    def decide_move(self, scent_grid=None):
        """
        mode == "return"        : 沿 return_path 走回巢穴
        mode == "return_to_food": 沿 return_path 走回最近一次的食物產地
        mode == "explore"       : 聞氣味 / 探索未知格 / 隨機亂走
        """
        # 回巢中：沿著規劃好的路走
        if self.mode == "return":
            if not self.return_path:
                return (0, 0)
            target = self.return_path.pop(0)
            dx = target[0] - self.pos[0]
            dy = target[1] - self.pos[1]
            return (dx, dy)

        # 回食物點：同樣沿著 return_path 走
        if self.mode == "return_to_food":
            if not self.return_path:
                # 路徑走完或失效，改成探索
                self.mode = "explore"
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
        """探索久了就會想回家。"""
        return self.mode == "explore" and self.steps_taken >= self.max_steps

    def reset_steps(self):
        self.steps_taken = 0

    def known_food_locations(self):
        return list(zip(*np.where(self.memory == 2)))

    def plan_return_path(self, nest_coords):
        """
        從自己的記憶中，用 BFS 找出回巢路徑，盡量不穿牆。
        nest_coords: 巢穴所有格子的座標 list[(x, y), ...]
        """
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

    def plan_path_to_food(self):
        """
        用 BFS 規劃一條路回到最近一次的食物產地（last_food_cluster）。
        走過的格子一樣不能是 memory == 3 的牆。
        """
        if self.last_food_cluster is None:
            return False

        target = self.last_food_cluster
        queue = deque()
        visited = set()
        start = tuple(self.pos)
        queue.append((start, [start]))
        visited.add(start)

        def walkable(cell):
            x, y = cell
            return self.memory[x][y] != 3

        while queue:
            current, path = queue.popleft()
            if current == target:
                self.return_path = path[1:]
                self.mode = "return_to_food"
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

        return False

    # ------------------------------------------------------------------ #
    # 食物區塊標記（維持原本功能）
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
