import pygame
import sys
from env_interface_2 import AntSimInterface
import random

COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_NEST = (100, 200, 255)
COLOR_FOOD = (0, 255, 100)
COLOR_ANT = (255, 255, 0)
COLOR_CARRYING = (255, 100, 100)
COLOR_MEMORY = (60, 60, 60)
COLOR_QUEEN = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_VISITED = (80, 80, 80)

CELL_SIZE = 5
MAP_SIZE = 150
WINDOW_SIZE = MAP_SIZE * CELL_SIZE

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 40))
pygame.display.set_caption("AntWorld Simulation Stage 2")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

sim = AntSimInterface(seed=random)


def draw():
    grid, ant_layer = sim.get_state()

    # 🐜 畫螞蟻走過的痕跡（灰色）在最底層
    for agent in sim.agents:
        if hasattr(agent, 'path_history'):
            for (x, y) in agent.path_history:
                rect = pygame.Rect(y * CELL_SIZE, x *
                                   CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, COLOR_VISITED, rect)

    # 畫主格子內容（背景/巢/食物）與螞蟻
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)

            if sim.nest_memory.is_explored(x, y):
                pygame.draw.rect(screen, COLOR_MEMORY, rect)

            if grid[x][y] == 1:
                color = COLOR_NEST
            elif grid[x][y] == 2:
                color = COLOR_FOOD
            else:
                color = COLOR_BG

            pygame.draw.rect(screen, color, rect)

            if ant_layer[x][y] == 3:
                pygame.draw.rect(screen, COLOR_CARRYING, rect)
            elif ant_layer[x][y] == 4:
                pygame.draw.rect(screen, COLOR_ANT, rect)

    # 畫蟻后
    qx, qy = sim.queen_pos
    q_rect = pygame.Rect(qy * CELL_SIZE, qx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, COLOR_QUEEN, q_rect)


def draw_info():
    text_area = pygame.Rect(0, WINDOW_SIZE, WINDOW_SIZE, 40)
    pygame.draw.rect(screen, (10, 10, 10), text_area)

    total = sim.food_delivered
    carrying = sum(1 for a in sim.agents if a.carrying_food)
    remaining = (sim.grid == 2).sum()
    ants = len(sim.agents)
    done = sum(1 for a in sim.agents if a.mode == "done")

    info = f"""🍃 Remaining: {remaining}   📦 Delivered: {total}   🎒 Carrying: {carrying}   🐜 Total: {ants}   ✅ Done: {done}"""
    txt_surface = font.render(info, True, COLOR_TEXT)
    screen.blit(txt_surface, (10, WINDOW_SIZE + 10))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    sim.step()
    screen.fill(COLOR_GRID)
    draw()
    draw_info()
    pygame.display.flip()
    clock.tick(144)
