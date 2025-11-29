import numpy as np
from envs.Adam_ants_2 import AntWorldEnv
from antagent.AntAgent import AntAgent


class AntSimInterface:
    def __init__(self, size=150, seed=None):
        self.size = size
        self.env = AntWorldEnv(size=size, seed=seed)
        self.grid = self.env.get_grid()
        self.agents = []
        self.agent_positions = {}  # (x,y): id
        self.tick = 0

        self.nest_coords = self._get_nest_coords()
        self.queen_pos = self._place_queen()
        self.food_delivered = 0

        self._init_agents()

    def _get_nest_coords(self):
        coords = []
        nx, ny = self.env.nest_pos
        for i in range(nx, nx + self.env.nest_size):
            for j in range(ny, ny + self.env.nest_size):
                coords.append((i, j))
        return coords

    def _place_queen(self):
        # 放在蟻窩中心
        nx, ny = self.env.nest_pos
        cx = nx + self.env.nest_size // 2
        cy = ny + self.env.nest_size // 2
        return (cx, cy)

    def _init_agents(self, total=16):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 1 and (i, j) not in self.agent_positions:
                    is_explorer = count < total // 2
                    agent = AntAgent(agent_id=count, pos=[
                                     i, j], is_explorer=is_explorer)
                    self.agents.append(agent)
                    self.agent_positions[(i, j)] = count
                    count += 1
                    if count >= total:
                        return

    def step(self):
        self.tick += 1
        self.agent_positions = {}

        for agent in self.agents:
            if agent.mode == "done":
                continue

            agent.observe(self.grid)

            if agent.should_return():
                agent.plan_return_path(self.nest_coords)

            dx, dy = agent.decide_move()
            moved = agent.move((dx, dy), self.grid)

            x, y = agent.pos
            if self.grid[x][y] == 2 and not agent.carrying_food:
                agent.carrying_food = True
                self.grid[x][y] = 0

            if tuple(agent.pos) in self.nest_coords:
                if agent.carrying_food:
                    agent.carrying_food = False
                    self.food_delivered += 1
                agent.mode = "done"

            if moved:
                self.agent_positions[(x, y)] = agent.id

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
