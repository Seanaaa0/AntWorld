import numpy as np
import random

from envs.Adam_ants_2 import AntWorldEnv
from antagent.AntAgent3 import AntAgent3


class NestMemory:
    """巢穴的全域記憶：哪些格子被探索過、哪裡有食物。"""

    def __init__(self, size=150):
        self.size = size
        self.explored = np.zeros((size, size), dtype=np.int8)  # 1: 探索過
        self.food_locs = set()

    def update_from_agent(self, agent: AntAgent3):
        mem = agent.memory
        self.explored |= (mem > 0).astype(np.int8)
        self.food_locs |= set(map(tuple, np.argwhere(mem == 2)))

    def get_known_food(self):
        return list(self.food_locs)

    def is_explored(self, x, y):
        return self.explored[x][y] == 1


class AntSimInterface:
    """
    與視覺化互動用的介面：
    - 包裝 AntWorldEnv
    - 管螞蟻群的 step / 狀態輸出
    """

    def __init__(self, size=150, seed=None):
        self.size = size
        self.env = AntWorldEnv(size=size, seed=seed)
        self.grid = self.env.get_grid()
        self.tick = 0
        self.scent_dirty = True

        self.nest_coords = self._get_nest_coords()
        self.queen_pos = self._place_queen()

        self.agents = []
        self.agent_positions = {}
        self.food_delivered = 0
        self.nest_memory = NestMemory(size)

        # 食物氣味場
        self.food_scent = np.zeros((size, size), dtype=float)
        self.xx, self.yy = np.meshgrid(
            np.arange(size), np.arange(size), indexing="ij"
        )
        self._init_agents()

        # 🐜 生小螞蟻相關
        self.max_ants = 300
        self.ticks_since_spawn = 0          # 距離上次生小螞蟻過了幾步
        self.food_since_spawn = 0           # 這段期間搬回來的食物數
        self.initial_nest_size = self.env.nest_size

        self._init_agents()
        self.BLOCK_LIMIT = 50  # 卡住太久就傳回巢穴重來
        self._update_food_scent()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def _get_nest_coords(self):
        coords = []
        nx, ny = self.env.nest_pos
        for i in range(nx, nx + self.env.nest_size):
            for j in range(ny, ny + self.env.nest_size):
                coords.append((i, j))
        return coords

    def _place_queen(self):
        nx, ny = self.env.nest_pos
        return (nx + self.env.nest_size // 2, ny + self.env.nest_size // 2)

    def _init_agents(self, total=16):
        """在巢穴內生成 16 隻螞蟻，一半探索、一半守巢。"""
        explorer_target = total // 2
        explorer_count = 0

        nest_spots = list(self.nest_coords)
        random.shuffle(nest_spots)

        for pos in nest_spots:
            if pos not in self.agent_positions:
                is_explorer = explorer_count < explorer_target
                agent = AntAgent3(
                    agent_id=len(self.agents),
                    pos=list(pos),
                    is_explorer=is_explorer,
                    size=self.size,
                )
                self.agents.append(agent)
                self.agent_positions[pos] = agent.id

                if is_explorer:
                    explorer_count += 1
                if len(self.agents) >= total:
                    break

    def _resize_nest(self):
        """
        根據目前螞蟻數量調整巢穴大小：
        - 巢穴是從原本 nest_pos 開始的正方形
        - 邊長 side ≈ ceil(sqrt(目前螞蟻數))
        - 不會比原本 env.nest_size 還小
        """
        n_ants = len(self.agents)
        base = self.initial_nest_size
        # 例如 16 ->4, 25->5, 100->10
        side = max(base, int(np.ceil(n_ants ** 0.5)))

        nx, ny = self.env.nest_pos

        new_coords = []
        for i in range(nx, nx + side):
            for j in range(ny, ny + side):
                if 0 <= i < self.size and 0 <= j < self.size:
                    new_coords.append((i, j))
                    # 把這些格子標成 1 (巢穴)，但在移動邏輯裡巢穴已被視為可穿透
                    self.grid[i][j] = 1

        self.nest_coords = new_coords
        # 重新計算蟻后位置
        self.queen_pos = (nx + side // 2, ny + side // 2)

    def _maybe_spawn_ant(self):
        """
        生小螞蟻規則：
        - 螞蟻數 < self.max_ants
        - 距離上次生產 >= 300 tick
        - 這段期間有至少 1 份食物被搬回巢穴
        """
        if len(self.agents) >= self.max_ants:
            return

        if self.ticks_since_spawn < 50:
            return

        if self.food_since_spawn <= 0:
            return

        # 在巢穴任一格生成新螞蟻，先全部當探索蟻
        spawn_pos = random.choice(self.nest_coords)
        new_ant = AntAgent3(
            agent_id=len(self.agents),
            pos=list(spawn_pos),
            is_explorer=True,
            size=self.size,
        )
        self.agents.append(new_ant)
        self.agent_positions[tuple(spawn_pos)] = new_ant.id

        # 重置計數
        self.ticks_since_spawn = 0
        self.food_since_spawn = 0

        # 巢穴跟著變大
        self._resize_nest()

    # ------------------------------------------------------------------ #
    # 食物氣味場：由目前所有食物重算
    # ------------------------------------------------------------------ #
    def _update_food_scent(self):
        # 如果沒有變化就不用重算
        if not self.scent_dirty:
            return

        self.scent_dirty = False
        R = 10
        self.food_scent.fill(0.0)
        food_positions = np.argwhere(self.grid == 2)
        if food_positions.size == 0:
            return

        dmin = np.full((self.size, self.size), R + 1, dtype=np.int16)
        for fx, fy in food_positions:
            dist = np.abs(self.xx - fx) + np.abs(self.yy - fy)
            np.minimum(dmin, dist, out=dmin)

        mask = dmin <= R
        self.food_scent[mask] = (R - dmin[mask]).astype(float)

        # 在距離 <= R 的地方給氣味，越近越濃
        mask = dmin <= R
        self.food_scent[mask] = (R - dmin[mask]).astype(float)

    # ------------------------------------------------------------------ #
    # 主迴圈
    # ------------------------------------------------------------------ #
    def step(self):
        self.tick += 1
        self.ticks_since_spawn += 1
        self.agent_positions = {}

        # 1) 先讓每隻螞蟻決定「想走哪」
        proposed_moves = {}
        for agent in self.agents:
            if agent.mode == "done":
                continue

            agent.observe(self.grid)

            # 探索模式下走太久就規劃回巢路
            if agent.should_return() and agent.mode == "explore":
                success = agent.plan_return_path(self.nest_coords)
                if not success:
                    agent.reset_steps()

            # 回巢模式但目前沒有路，就再試一次
            if agent.mode == "return" and not agent.return_path:
                agent.plan_return_path(self.nest_coords)

            dx, dy = agent.decide_move(self.food_scent)
            proposed_moves[agent.id] = (dx, dy)

        # 2) 根據提案實際移動，處理牆 / 碰撞 / 回巢 / 撿食物
        new_positions = {}

        for agent in self.agents:
            if agent.mode == "done":
                continue

            dx, dy = proposed_moves.get(agent.id, (0, 0))
            new_x = agent.pos[0] + dx
            new_y = agent.pos[1] + dy

            moved = False

            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                target = (new_x, new_y)
                is_nest = target in self.nest_coords
                # 只有「非巢穴的 1」才當牆，巢穴是可以穿透的
                is_wall = (self.grid[new_x][new_y] == 1) and (not is_nest)
                # 巢穴格允許多隻螞蟻疊在一起
                occupied = (target in new_positions) and (not is_nest)

                if (not is_wall) and (not occupied):
                    agent.pos = [new_x, new_y]
                    agent.steps_taken += 1
                    agent.path_history.append(target)
                    new_positions[target] = agent.id
                    agent.blocked_count = 0
                    moved = True

            if not moved:
                agent.blocked_count += 1
                if agent.blocked_count >= self.BLOCK_LIMIT:
                    # 卡太久：傳回巢穴重來
                    spawn = random.choice(self.nest_coords)
                    agent.pos = list(spawn)
                    agent.carrying_food = False
                    agent.mode = "explore"
                    agent.return_path = []
                    agent.reset_steps()
                    agent.blocked_count = 0
                    agent.path_history.append(spawn)

            x, y = agent.pos

            # 撿到食物
            if self.grid[x][y] == 2 and not agent.carrying_food:
                agent.carrying_food = True
                self.grid[x][y] = 0
                self.scent_dirty = True  # ✅ 告訴氣味系統需要重算
                agent.mark_food_region((x, y), self.grid)
                agent.plan_return_path(self.nest_coords)

            # 回到巢穴任一格

            if (x, y) in self.nest_coords:
                # 有食物就放下
                if agent.carrying_food:
                    agent.carrying_food = False
                    self.food_delivered += 1
                    self.food_since_spawn += 1      # ✅ 這段期間多搬回一份食物

                # 巢內整合記憶
                self.nest_memory.update_from_agent(agent)

                if agent.is_explorer:
                    # 女王把目前知道的食物位置灑回給這隻螞蟻
                    for fx, fy in self.nest_memory.get_known_food():
                        agent.memory[fx, fy] = 2

                    # 探索蟻：重設狀態繼續探索
                    agent.mode = "explore"
                    agent.reset_steps()
                    agent.return_path = []
                    agent.blocked_count = 0
                else:
                    # 守巢蟻就留在巢內
                    agent.mode = "done"

        self.agent_positions = new_positions

        # 3) 依據最新 grid 狀態重算食物氣味
        self._update_food_scent()

        # 4) 看看要不要生小螞蟻
        self._maybe_spawn_ant()

    # ------------------------------------------------------------------ #
    # 輸出狀態給視覺化
    # ------------------------------------------------------------------ #
    def get_state(self):
        grid_copy = self.grid.copy()
        ant_layer = np.zeros_like(grid_copy)

        for agent in self.agents:
            if agent.mode == "done":
                continue
            x, y = agent.pos
            ant_layer[x][y] = 3 if agent.carrying_food else 4

        return grid_copy, ant_layer

    def is_done(self):
        # 當螞蟻總數 >= max_ants 就視為結束
        return len(self.agents) >= self.max_ants
