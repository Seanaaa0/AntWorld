import numpy as np
import random

from envs.Adam_ants_2 import AntWorldEnv
from antagent.AntAgent4 import AntAgent4


class NestMemory:
    """巢穴的全域記憶：哪些格子被探索過、哪裡有食物。"""

    def __init__(self, size=150):
        self.size = size
        self.explored = np.zeros((size, size), dtype=np.int8)  # 1: 探索過
        self.food_locs = set()

    def update_from_agent(self, agent: AntAgent4):
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

        self.nest_coords = self._get_nest_coords()
        self.queen_pos = self._place_queen()

        self.agents = []
        self.agent_positions = {}
        self.food_delivered = 0
        self.nest_memory = NestMemory(size)

        # 食物氣味場（向量化運算用）
        self.food_scent = np.zeros((size, size), dtype=float)
        self.xx, self.yy = np.meshgrid(
            np.arange(size), np.arange(size), indexing="ij"
        )
        self.scent_dirty = True  # 食物變化時才重算氣味

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
                agent = AntAgent4(
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

    # ------------------------------------------------------------------ #
    # 食物氣味場：由目前所有食物重算（向量化）
    # ------------------------------------------------------------------ #
    def _update_food_scent(self):
        """
        scent(x, y) = max(0, R - min_food |x - fx| + |y - fy|)
        這裡 R = 10，代表氣味範圍是曼哈頓距離 <= 10。
        只有在 self.scent_dirty 為 True 時才重算。
        """
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

    # ------------------------------------------------------------------ #
    # 螞蟻間 5x5 區域內分享食物產地記憶
    # ------------------------------------------------------------------ #
    def _share_food_clusters(self):
        """
        若兩隻螞蟻距離 <= 2（5x5 區域），
        則交換各自的 last_food_cluster（取 timestamp 較新的那個）。
        """
        active = [a for a in self.agents if a.mode != "done"]
        n = len(active)
        for i in range(n):
            a = active[i]
            for j in range(i + 1, n):
                b = active[j]
                if abs(a.pos[0] - b.pos[0]) <= 2 and abs(a.pos[1] - b.pos[1]) <= 2:
                    # 如果兩邊都沒有資訊就略過
                    if a.last_food_cluster is None and b.last_food_cluster is None:
                        continue

                    # a 的資訊比較新
                    if (
                        a.last_food_cluster is not None
                        and (
                            b.last_food_cluster is None
                            or a.last_food_timestamp > b.last_food_timestamp
                        )
                    ):
                        b.last_food_cluster = a.last_food_cluster
                        b.last_food_timestamp = a.last_food_timestamp

                    # b 的資訊比較新
                    elif (
                        b.last_food_cluster is not None
                        and (
                            a.last_food_cluster is None
                            or b.last_food_timestamp > a.last_food_timestamp
                        )
                    ):
                        a.last_food_cluster = b.last_food_cluster
                        a.last_food_timestamp = b.last_food_timestamp

    # ------------------------------------------------------------------ #
    # 判斷「回到原食物點」是否已抵達 & 沒氣味 → 切換成探索
    # ------------------------------------------------------------------ #
    def _handle_arrived_food_cluster(self):
        for agent in self.agents:
            if agent.mode != "return_to_food":
                continue
            if agent.last_food_cluster is None:
                continue

            lx, ly = agent.last_food_cluster
            ax, ay = agent.pos
            dist = abs(ax - lx) + abs(ay - ly)

            # 抵達原食物區附近（半徑 2 以內）
            if dist <= 2:
                # 若這一帶已經沒有氣味了，就改為從此處繼續探索
                if self.food_scent[ax, ay] <= 0.0:
                    agent.mode = "explore"
                    agent.return_path = []
                    agent.reset_steps()
                    agent.blocked_count = 0

    # ------------------------------------------------------------------ #
    # 主迴圈
    # ------------------------------------------------------------------ #
    def step(self):
        self.tick += 1
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

            # 撿到食物：記錄「這次的食物產地」，並準備回巢
            if self.grid[x][y] == 2 and not agent.carrying_food:
                agent.carrying_food = True
                self.grid[x][y] = 0
                self.scent_dirty = True  # 食物改變，需要重算氣味場

                # 記住這次的食物產地位置（用單一格代表整個 cluster）
                agent.last_food_cluster = (x, y)
                agent.last_food_timestamp = self.tick

                agent.mark_food_region((x, y), self.grid)
                agent.plan_return_path(self.nest_coords)

            # 回到巢穴任一格
            if (x, y) in self.nest_coords:
                just_delivered = False

                # 有食物就放下
                if agent.carrying_food:
                    agent.carrying_food = False
                    self.food_delivered += 1
                    just_delivered = True

                # 巢內整合記憶
                self.nest_memory.update_from_agent(agent)

                if agent.is_explorer:
                    # 女王把目前知道的食物位置灑回給這隻螞蟻
                    for fx, fy in self.nest_memory.get_known_food():
                        agent.memory[fx, fy] = 2

                    # 若是剛剛搬完食物回巢，就優先回到那個食物產地
                    if just_delivered and agent.last_food_cluster is not None:
                        planned = agent.plan_path_to_food()
                        if not planned:
                            agent.mode = "explore"
                    else:
                        # 其他情況：從巢穴出發繼續探索
                        if agent.mode == "return":
                            agent.mode = "explore"

                    agent.reset_steps()
                    agent.blocked_count = 0

                else:
                    # 守巢蟻就留在巢內
                    agent.mode = "done"

        self.agent_positions = new_positions

        # 3) 依據最新 grid 狀態重算食物氣味（若有需要）
        self._update_food_scent()

        # 4) 若有螞蟻回到原食物區但已無氣味，改成從那裡繼續探索
        self._handle_arrived_food_cluster()

        # 5) 螞蟻之間在 5x5 區域內分享「最近食物產地」記憶
        self._share_food_clusters()

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
        # 先維持原本：運完 100 單位食物就當作結束
        return self.food_delivered >= 100
