import numpy as np
import random

from envs.Adam_ants_2 import AntWorldEnv
from antagent.AntAgent2 import AntAgent as AntAgent2


class NestMemory:
    """巢穴的全域記憶：哪些格子被探索過、哪裡有食物。"""

    def __init__(self, size=150):
        self.size = size
        self.explored = np.zeros((size, size), dtype=np.int8)  # 1: 探索過
        self.food_locs = set()

    def update_from_agent(self, agent: AntAgent2):
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

        self._init_agents()
        self.BLOCK_LIMIT = 50  # 卡住太久就傳回巢穴重來

    # --------------------------------------------------------------------- #
    # 初始化
    # --------------------------------------------------------------------- #
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
                agent = AntAgent2(
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

    # --------------------------------------------------------------------- #
    # 工具函式
    # --------------------------------------------------------------------- #
    def _kick_from_nest(self, agent: AntAgent2):
        """
        把回巢的探索蟻「踢」出巢穴一格，
        避免待在巢穴格裡面不動。
        """
        x, y = agent.pos
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                target = (nx, ny)
                # 不要再留在巢穴內，也不要撞牆
                if self.grid[nx][ny] != 1 and target not in self.nest_coords:
                    agent.pos = [nx, ny]
                    agent.path_history.append(target)
                    return

        # 四周都是巢或牆就先算了，不強制

    # --------------------------------------------------------------------- #
    # 主迴圈
    # --------------------------------------------------------------------- #
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

            dx, dy = agent.decide_move()
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
                is_wall = (self.grid[new_x][new_y] == 1) and (not is_nest)
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
                agent.mark_food_region((x, y), self.grid)
                agent.plan_return_path(self.nest_coords)

                # 回到巢穴：只處理「放下食物」與「更新記憶」，不要瞬移
            if (x, y) in self.nest_coords:

                # 有食物就放下，計數 +1
                if agent.carrying_food:
                    agent.carrying_food = False
                    self.food_delivered += 1

                # 把個人記憶整合進巢穴記憶
                self.nest_memory.update_from_agent(agent)

                # 重設步數，避免立刻又覺得「該回家」
                agent.reset_steps()

                # 清掉回家的路線，讓他下一步用探索邏輯決定去哪
                agent.return_path = []
                agent.mode = "explore"
                agent.blocked_count = 0

                # 不要改 agent.pos，不要瞬移，不要踢出去

        self.agent_positions = new_positions

    # --------------------------------------------------------------------- #
    # 輸出狀態給視覺化
    # --------------------------------------------------------------------- #
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
        # 先隨便設一個：運完 100 單位食物就當作結束
        return self.food_delivered >= 100
