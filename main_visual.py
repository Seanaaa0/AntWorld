import pygame
import sys
from env_interface import AntSimInterface

# 顏色定義
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_NEST = (100, 200, 255)
COLOR_FOOD = (0, 255, 100)
COLOR_ANT = (255, 255, 0)
COLOR_CARRYING = (255, 100, 100)
COLOR_TEXT = (255, 255, 255)

CELL_SIZE = 5
MAP_SIZE = 150
WINDOW_SIZE = MAP_SIZE * CELL_SIZE

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 40))  # 下方加文字欄
pygame.display.set_caption("AntWorld Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

sim = AntSimInterface(seed=42)

def draw():
    grid, ant_layer = sim.get_state()
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if grid[x][y] == 1:
                color = COLOR_NEST
            elif grid[x][y] == 2:
                color = COLOR_FOOD
            else:
                color = COLOR_BG

            pygame.draw.rect(screen, color, rect)

            # 螞蟻在上面
            if ant_layer[x][y] == 3:
                pygame.draw.rect(screen, COLOR_CARRYING, rect)
            elif ant_layer[x][y] == 4:
                pygame.draw.rect(screen, COLOR_ANT, rect)

def draw_info():
    text_area = pygame.Rect(0, WINDOW_SIZE, WINDOW_SIZE, 40)
    pygame.draw.rect(screen, (10, 10, 10), text_area)

    total = sim.food_delivered
    carrying = sum(1 for a in sim.agents if a.carrying_food)
    remaining = (sim.grid == 2).sum()
    ants = len(sim.agents)

    info = f"🍃 Remaining: {remaining}   📦 Delivered: {total}   🎒 Carrying: {carrying}   🐜 Total ants: {ants}"
    txt_surface = font.render(info, True, COLOR_TEXT)
    screen.blit(txt_surface, (10, WINDOW_SIZE + 10))

# 主迴圈
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
    clock.tick(10)