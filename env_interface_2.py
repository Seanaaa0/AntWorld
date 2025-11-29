import numpy as np
from envs.Adam_ants_2 import AntWorldEnv
from antagent.AntAgent import AntAgent
import random


class NestMemory:
    def __init__(self, size=150):
        self.size = size
        self.explored = np.zeros((size, size), dtype=np.int8)  # 1: 探索過
        self.food_locs = set()

    def update_from_agent(self, agent):
        mem = agent.memory
        self.explored |= (mem > 0).astype(np.int8)
        self.food_locs |= set([tuple(p) for p in np.argwhere(mem == 2)])

    def get_known_food(self):
        return list(self.food_locs)

    def is_explored(self, x, y):
        return self.explored[x][y] == 1


class AntSimInterface:
    def __init__(self, size=150, seed=None):
        self.size = size
        self.env = AntWorldEnv(size=size, seed=seed)
        self.grid = self.env.get_grid()
        self.agents = []
        self.agent_positions = {}
        self.tick = 0
        self.departure_queue = []  # 裝螞蟻 id
        self.departure_index = 0  # 每輪最多讓一隻出巢

        self.nest_coords = self._get_nest_coords()
        self.queen_pos = self._place_queen()
        self.food_delivered = 0
        self.nest_memory = NestMemory(size)

        self._init_agents()
        self.departure_queue = [a.id for a in self.agents if a.is_explorer]

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
        explorer_target = total // 2
        explorer_count = 0

        nest_spots = list(self._get_nest_coords())
        random.shuffle(nest_spots)

        for pos in nest_spots:
            if pos not in self.agent_positions:
                is_explorer = explorer_count < explorer_target
                agent = AntAgent(
                    agent_id=len(self.agents),
                    pos=list(pos),
                    is_explorer=is_explorer
                )
                self.agents.append(agent)
                self.agent_positions[pos] = agent.id
                if is_explorer:
                    explorer_count += 1
                if len(self.agents) >= total:
                    break

        # if explorer_count < explorer_target:
        #     print(
        #         f"[⚠️ 警告] 實際探索蟻數量僅為 {explorer_count}，少於預期 {explorer_target}。")

    def step(self):
        self.tick += 1
        self.agent_positions = {}

        # 第一步：決定所有 agent 要去哪
        proposed_moves = {}
        for agent in self.agents:
            if agent.mode == "done":
                continue

            agent.observe(self.grid)

            if agent.should_return() and agent.mode == "explore":
                success = agent.plan_return_path(self.nest_coords)
                if not success:
                    agent.reset_steps()
                    print(f"[{agent.id}] 無法規劃回巢路，繼續探索！")

            if agent.mode == "return" and not agent.return_path:
                agent.plan_return_path(self.nest_coords)

            if hasattr(agent, 'just_reset') and agent.just_reset:
                proposed_moves[agent.id] = (0, 0)
            else:
                dx, dy = agent.decide_move()
                proposed_moves[agent.id] = (dx, dy)

        # 第二步：根據 agent_positions 判斷誰可以移動
        new_positions = {}
        for agent in self.agents:
            if agent.mode == "done":
                continue

            dx, dy = proposed_moves.get(agent.id, (0, 0))
            new_x = agent.pos[0] + dx
            new_y = agent.pos[1] + dy

            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if self.grid[new_x][new_y] != 1 and (new_x, new_y) not in self.agent_positions:
                    agent.pos = [new_x, new_y]
                    agent.steps_taken += 1
                    agent.path_history.append((new_x, new_y))
                    new_positions[(new_x, new_y)] = agent.id
                else:
                    agent.blocked_count += 1
                    print(
                        f"[{agent.id}] 移動到 ({new_x},{new_y}) 被擋住（連續 {agent.blocked_count} 次）")
                    if agent.blocked_count >= 3:
                        print(f"[{agent.id}] 被卡 {agent.blocked_count} 次，強制切回探索")
                        agent.return_path = []
                        agent.mode = "explore"
                        agent.reset_steps()
            else:
                agent.blocked_count += 1

            x, y = agent.pos
            if self.grid[x][y] == 2 and not agent.carrying_food:
                agent.carrying_food = True
                self.grid[x][y] = 0
                agent.mark_food_region((x, y), self.grid)
                agent.plan_return_path(self.nest_coords)

            if tuple(agent.pos) in self.nest_coords:
                if agent.carrying_food:
                    agent.carrying_food = False
                    self.food_delivered += 1
                self.nest_memory.update_from_agent(agent)

                if agent.is_explorer and agent.mode == "return":
                    agent.mode = "explore"
                    agent.reset_steps()
                    agent.return_path = []
                    agent.just_reset = True
                    print(f"[{agent.id}] 回巢後重啟探索，從 {agent.pos} 出發")
                elif not agent.is_explorer:
                    agent.mode = "done"

            # 清除 just_reset 標記讓下回合能動
            if hasattr(agent, 'just_reset') and agent.just_reset:
                agent.just_reset = False

        # 控制探索蟻出發順序（每 5 tick 一隻）
        if hasattr(self, 'departure_queue') and hasattr(self, 'departure_index'):
            if self.tick % 5 == 0 and self.departure_index < len(self.departure_queue):
                i = self.departure_queue[self.departure_index]
                a = self.agents[i]
                if a.mode == "explore" and tuple(a.pos) in self.nest_coords:
                    print(f"[排程] 探索蟻 {i} 被允許出巢")
                    a.just_reset = False
                    self.departure_index += 1

        self.agent_positions = new_positions

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
        return self.food_delivered >= 100
