

class MyClass:
    def __init__(self):
        self.value = 0  # ✅ self работает, потому что это метод класса

    def set_value(self, value):
        self.value = value  # ✅ правильно используем self



from enum import Enum

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    GAME_OVER = 3

# Пример использования
current_state = GameState.MENU

if current_state == GameState.MENU:
    print("Показать меню")
elif current_state == GameState.PLAYING:
    print("Игра идёт")


import pygame
import sys
import random
import logging

# ---------------------------
# Логгер
# ---------------------------
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:  # чтобы обработчики не дублировались
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger

logger = get_logger(__name__)
logger.info("Логгер успешно создан!")

# ---------------------------
# Инициализация Pygame
# ---------------------------
pygame.init()
logger.info("Pygame успешно инициализирован!")

# Пример создания окна (можешь изменить размер)
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("My Pac-Man 1")

# ---------------------------
# Основной цикл игры
# ---------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            logger.info("Закрытие игры пользователем")
            running = False

pygame.quit()
logger.info("Игра завершена")

# Combined Python file generated from PyPacman-main (3).zip
# Do not edit unless you know what you are doing.


# ===== File: PyPacman-main/src/configs.py =====
class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WALL = (112, 167, 255)
    YELLOW = (252, 186, 3)
    WALL_BLUE = (24, 24, 217)


SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
CELL_SIZE = (20, 20)
NUM_ROWS = 31
NUM_COLS = 28
PACMAN = (32, 32)
GHOSTS = (32, 32)
PACMAN_SPEED = 4
GHOST_SPEED_FAST = 5
GHOST_SPEED_SLOW = 2
GHOST_NORMAL_DELAY = 5000

DOT_POINT = 10
POWER_POINT = 15
GHOST_POINT = 25
LEVEL_COMP_POINT = 80

GHOST_DELAYS = {
    "inky": 12000,
    "pinky": 8000,
    "blinky": 4000,
    "clyde": 16000,
    "blue": 0
}
GHOST_TARGET_CHANGE = {
    "inky": 10,
    "pinky": 8,
    "blinky": 6,
    "clyde": 7,
    "blue": 7
}
GHOST_SCATTER_TARGETS = {
    'blinky': (0, 30),
    "pinky": (0, 0),
    "inky": (31, 0),
    "clyde": (31, 30)
}
loading_screen_gif = "assets/other/loading.gif"
# ===== File: PyPacman-main/src/sprites/sprite_configs.py =====
PACMAN_PATHS = {
    "left": [
        "assets/pacman-left/1.png",
        "assets/pacman-left/2.png",
        "assets/pacman-left/3.png",
    ],
    "right": [
        "assets/pacman-right/1.png",
        "assets/pacman-right/2.png",
        "assets/pacman-right/3.png",
    ],
    "up": [
        "assets/pacman-up/1.png",
        "assets/pacman-up/2.png",
        "assets/pacman-up/3.png",
    ],
    "down": [
        "assets/pacman-down/1.png",
        "assets/pacman-down/2.png",
        "assets/pacman-down/3.png",
    ],
}

GHOST_PATHS = {
    "inky": ["assets/ghosts/inky.png"],
    "pinky": ["assets/ghosts/pinky.png"],
    "blinky": ["assets/ghosts/blinky.png"],
    "clyde": ["assets/ghosts/clyde.png"],
    "blue": ["assets/ghosts/blue_ghost.png"]
}


# ===== File: PyPacman-main/src/utils/coord_utils.py =====
def center_element(screen_width, screen_height, element_width, element_height):
    return place_elements_offset(
        screen_width, screen_height, element_width, element_height, 0.5, 0.6
    )


def place_elements_offset(
    screen_width, screen_height, element_width, element_height, xoffset, yoffset
):
    x = (screen_width - element_width) * xoffset
    y = (screen_height - element_height) * yoffset
    return x, y


def __get_x_y(pos, num_rows, num_cols):
    x = pos[0]
    y = pos[1]
    if pos[0] < 0:
        x = num_rows + x
    if pos[1] < 0:
        y = num_cols + y
    return x, y


def get_coords_from_idx(
    pacman_pos, start_x, start_y, cell_w, cell_h, num_rows, num_cols
):
    x, y = __get_x_y(pacman_pos, num_rows, num_cols)
    x_coord = start_x + (y * cell_w)
    y_coord = start_y + (x * cell_h)
    return x_coord, y_coord


def precompute_matrix_coords(start_x, start_y, cell_size, num_rows, num_cols):
    matrix_coords = []
    col_start = start_y
    for _ in range(num_rows):
        row_start = start_x
        m = []
        for _ in range(num_cols):
            m.append([row_start, col_start])
            row_start += cell_size
        col_start += cell_size
        matrix_coords.append(m)
    return matrix_coords


def get_idx_from_coords(x_coord, y_coord, start_x, start_y, cell_size):
    x_pos = int((x_coord - start_x) // cell_size)
    y_pos = int((y_coord - start_y) // cell_size)
    return y_pos, x_pos  # in matrix, horizontal is columns and vertical are rows

def get_tiny_matrix(matrix, cell_size, pacman_speed):
    sub_div = cell_size // pacman_speed
    num_rows = len(matrix) * sub_div
    num_cols = len(matrix[0]) * sub_div
    tiny_matrix = [["null"] * num_cols for _ in range(num_rows)]
    tiny_r, tiny_c = 0, 0
    for row in matrix:
        for cell in row:
            if cell != "wall":
                cell = "null"
            for sx in range(sub_div):
                for sy in range(sub_div):
                    tiny_matrix[tiny_r + sx][tiny_c + sy] = cell
            tiny_c += sub_div
        tiny_r += sub_div
        tiny_c = 0
    return tiny_matrix

def get_movable_locations(matrix, max_cell_size=20, 
                          cell_size=20):
    movables = []
    rows, cols = len(matrix), len(matrix[0])  # Matrix dimensions
    subdiv = max_cell_size // cell_size

    def is_free(r, c):
        return 0 <= r < rows and 0 <= c < cols and matrix[r][c] not in ('wall', 'elec')
    
    def is_valid(r, c):
        for x in range(subdiv*2):
            for y in range(subdiv*2):
                if not is_free(r+x, c+y):
                    return False
        return True

    for r_idx in range(rows):
        for c_idx in range(cols):
            if (r_idx + (subdiv*2) <= rows and \
                 c_idx + (subdiv*2) <= cols) and is_valid(r_idx, c_idx):
                movables.append((r_idx, c_idx))

    return movables

def is_any_wall(matrix, x, y):
    rows, cols = len(matrix), len(matrix[0])  # Matrix dimensions

    def is_wall(r, c):
        """Check if the cell is 'wall' and within bounds."""
        return 0 <= r < rows and 0 <= c < cols and matrix[r][c] == 'wall'

    # Check all four positions
    return (
        is_wall(x, y) or
        is_wall(x, y + 1) or
        is_wall(x + 1, y + 1) or
        is_wall(x + 1, y)
    )
# ===== File: PyPacman-main/src/utils/draw_utils.py =====
from pygame import draw


def draw_rect(x, y, w, h, screen, color, fill=0):
    draw.rect(screen, color, (x, y, w, h), fill)


def draw_circle(x, y, radius, screen, color):
    draw.circle(screen, color, (x, y), radius)


def draw_debug_rects(start_x, start_y, num_rows, num_cols, cell_size, color, screen):
    curr_x, curr_y = start_x, start_y
    for _ in range(num_rows):
        for _ in range(num_cols):
            draw.rect(screen, color, (curr_x, curr_y, cell_size, cell_size), 1)
            curr_x += cell_size
        curr_y += cell_size
        curr_x = start_x

# ===== File: PyPacman-main/src/utils/ghost_movement_utils.py =====
import math

DIRECTION_MAPPER = {"up":[(-1, 0), (-1, 1)],
                    "left":[(0, -1), (1, -1)],
                    "down":[(2, 0), (2, 1)],
                    "right":[(0, 2), (1, 2)]}
BLOCKERS = ['wall', 'elec']

def get_is_move_valid(curr_pos, direction, matrix):
    next_indices = DIRECTION_MAPPER[direction]
    for r, c in next_indices:
        next_c = curr_pos[1] + c
        if next_c < 0 or next_c >= len(matrix[0]): #because there is only 1 place where ghost can go out of bounds.
            continue
        if matrix[curr_pos[0] + r][curr_pos[1] + c] in BLOCKERS:
            return False
    return True

def eucliad_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_direction(ghost_matrix_pos: tuple[int, int],
                target_matrix_pos: tuple[int, int],
                matrix: list[list[str]],
                prev: tuple[int, int]):
    g1, g2 = ghost_matrix_pos
    t1, t2 = target_matrix_pos
    num_rows, _ = len(matrix), len(matrix[0])
    next_direction_mapper = {"up":(-1, 0), "down": (1, 0), "left":(0, -1), "right": (0, 1)}
    directions = ['up', 'left', 'down','right']
    curr_min = float('inf')
    target_dir = None
    for direction in directions:
        is_movable = get_is_move_valid((g1, g2), direction, matrix)
        if not is_movable:
            continue
        direction_additives = next_direction_mapper[direction]
        next_x, next_y = g1 + direction_additives[0], g2 + direction_additives[1]
        if next_x >= num_rows or next_x < 0:
            continue
        if next_direction_mapper[direction] == prev:
            continue
        distance = eucliad_distance((next_x, next_y), (t1, t2))
        if distance < curr_min:
            curr_min = distance
            target_dir = direction
    # print(ghost_matrix_pos, matrix[ghost_matrix_pos[0]][ghost_matrix_pos[1]])
    # print(target_dir)
    if target_dir is None:
        print('error', prev, ghost_matrix_pos, matrix[ghost_matrix_pos[0]][ghost_matrix_pos[1]])
        raise ValueError("Oh my god, I don't know what to do, im crashing the game")
    return next_direction_mapper[target_dir]

def get_is_intersection(ghost_matrix_pos: tuple[int, int], 
                        matrix: list[list[str]],
                        prev=None):
    possible_moves = 0
    for k, _ in DIRECTION_MAPPER.items():
        if prev == k:
            continue
        if get_is_move_valid(ghost_matrix_pos, k, matrix):
            possible_moves += 1
    return possible_moves > 1



# ===== File: PyPacman-main/src/utils/graph_utils.py =====
import heapq

def a_star(matrix, start, target, subdivs=4):
    rows, cols = len(matrix), len(matrix[0])

    def is_valid(x, y):
        """Check if all cells in the subdivs x subdivs block are valid."""
        if not (0 <= x < rows and 0 <= y < cols):
            return False
        for dx in range(subdivs*2):
            for dy in range(subdivs*2):
                if x + dx >= rows or y + dy >= cols or matrix[x + dx][y + dy] == 'wall':
                    return False
        return True

    def heuristic(a, b):
        """Calculate Manhattan distance."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def path_builder(current, came_from):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]

    # Directions: Up, Down, Left, Right
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1)
    ]

    open_set = []
    heapq.heappush(open_set, (0, start))  # (priority, position)
    came_from = {}  # To reconstruct the path

    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}
    closest_node = start
    closest_distance = heuristic(start, target)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            # Reconstruct path
            return path_builder(current, came_from) #
            

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if is_valid(neighbor[0], neighbor[1]):
                tentative_g_score = g_score[current] + 1  # All moves cost 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)

                    if neighbor not in [pos for _, pos in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
                h_dist = heuristic(neighbor, target)
                if h_dist < closest_distance:
                    closest_node = neighbor
                    closest_distance = h_dist

    return path_builder(closest_node, came_from)  #
# ===== File: PyPacman-main/src/sprites/ghosts.py =====
"""
This module handles ghosts
A single ghost class responsible for rending ghost,
ghost manager class responsible for creating multiple ghost objects aka ghosts
ghose movement class responsible for moving the ghost with lerp
GhostManager takes ghost matrix pos, grid start pos, matrix, game state object.
Ghost takes name, ghost matrix pos, grid start pos, matrix and game state object
Ghost movement class should need ghost coordinate pos, matrix, game state, some more parameters.
"""
from abc import abstractmethod, ABC

from pygame.sprite import Sprite
from pygame import Surface
from pygame import image, transform
import pygame.time as pytime
from pygame.time import wait
from pygame.rect import Rect

import random


logger = get_logger(__name__)

class Ghost(Sprite, ABC):
    def __init__(self,
                 name: str,
                 ghost_matrix_pos: tuple[int, int],
                 grid_start_pos: tuple[int | float, int | float],
                 matrix: list[list[str]],
                 game_state: GameState
                 ):
        super().__init__()
        self.name = name
        self._ghost_matrix_pos = ghost_matrix_pos
        self._grid_start_pos = grid_start_pos
        self._matrix = matrix
        self.num_rows = len(self._matrix)
        self.num_cols = len(self._matrix[0])
        self._game_state = game_state
        self._is_released = False
        self._creation_time = pytime.get_ticks()
        self._dead_wait = GHOST_DELAYS[self.name]
        self.move_direction_mapper = {"up": (-1, 0), "down":(2, 0), 
                                      "right": (0, 2), "left": (0, -1)}
        self.prev_pos = None
        self._t = 0
        self._accelerate = 0.2
        self._direction = None
        self._target = None
        self.prev = None
        self.next_tile = None
        self._direction_prevent = {(-1, 0): (1, 0), (1, 0): (-1, 0),
                                     (0, 1): (0, -1), (0, -1): (0, 1)}
        self.is_scared = False
        self.curr_pos = None
        self.release_time = None
        self.sounds = SoundManager()
        self.load_images()

    def _get_coords_from_idx(self, p1):
        return get_coords_from_idx(p1, *self._grid_start_pos, 
                                   *CELL_SIZE, self.num_rows, self.num_cols)
    
    def _get_idx_from_coords(self, p1):
        return get_idx_from_coords(*p1,
                                   *self._grid_start_pos, CELL_SIZE[0])
    
    def build_bounding_boxes(self, x, y):
        self.rect.x = x + (CELL_SIZE[0] * 2 - self.rect.width) // 2
        self.rect.y = y + (CELL_SIZE[1] * 2 - self.rect.height) // 2

    def load_images(self):
        ghost_images = GHOST_PATHS[self.name][0]
        blue_images = GHOST_PATHS['blue'][0]
        self.normal_image = transform.scale(image.load(ghost_images).convert_alpha(),
                                     PACMAN)
        self.blue_image = transform.scale(image.load(blue_images).convert_alpha(),
                                     PACMAN)
        self.image = self.normal_image
        x, y = self._get_coords_from_idx(self._ghost_matrix_pos)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.rect_x = x
        self.rect_y = y

    def lerp(self, source, dest):
        x1, y1 = source
        x2, y2 = dest
        if self._target is None or self._t == 1:
            return x1, y1
        if self._t < 1:
            self._t += self._accelerate
        else:
            self._t = 1  

        x = (1 - self._t) * x1 + self._t * x2
        y = (1 - self._t) * y1 + self._t * y2
        return x, y
    
    def check_is_released(self):
        if self._is_released:
            return
        curr_time = pytime.get_ticks()
        if (curr_time - self._creation_time) > self._dead_wait:
            self._is_released = True
            self._dead_wait = 1500
            self.rect_x, self.rect_y = self._get_coords_from_idx((11, self._ghost_matrix_pos[1]))
            self.release_time = pytime.get_ticks()

    def move_ghost(self):
        if not self._is_released:
            return
        if self._target is None:
            self.prepare_movement()
        source = self._get_coords_from_idx(self.prev)
        dest = self._get_coords_from_idx(self.next_tile)
        self.rect_x, self.rect_y = self.lerp(source, dest)
        curr_mat_x, curr_mat_y = self._get_idx_from_coords((self.rect_x, self.rect_y))
        if self.name == 'blinky':
            self._game_state.blinky_matrix_pos = (curr_mat_x, curr_mat_y)
        self.curr_pos = (curr_mat_x, curr_mat_y)
        if (self._t == 1) \
            or (self.rect_x == dest[0] and self.rect_y == dest[1]):
            check_prev = self._direction_prevent.get(self._direction)
            prev_val = self._get_direction_reverse_map(check_prev)
            if get_is_intersection(self.next_tile, self._matrix, 
                                   prev_val):
                
                self.prepare_movement()
            else:
                if not get_is_move_valid(self.next_tile, 
                                         self._get_direction_reverse_map(self._direction), 
                                         self._matrix):
                    self.prepare_movement()
                else:
                    self.prev = self.next_tile
                    self.next_tile = (self.next_tile[0] + self._direction[0],
                                  self.next_tile[1] + self._direction[1])
                
                self._t = 0
    
    def _get_direction_reverse_map(self, direction):
        match direction:
            case (-1, 0):
                return 'up'
            case (1, 0):
                return "down"
            case (0, -1):
                return "left"
            case(0, 1):
                return "right"
      
    def _boundary_check(self):
        if not self.next_tile:
            return
        if (self.next_tile[1] >= self.num_cols):
            self.next_tile = (self.next_tile[0], 0)
            return
        if self.next_tile[1] < 0:
            self.next_tile = (self.next_tile[0], self.num_cols - 1)

    def prepare_movement(self):
        ghost_x, ghost_y = self._get_idx_from_coords((self.rect_x, self.rect_y))
        if self.next_tile:
            ghost_x, ghost_y = self.next_tile
        if self.is_scared:
            self._target = self.get_random_target()
        else:
            self._target = self.determine_target()
        prev = self._direction_prevent.get(self._direction)
        self._direction = get_direction((ghost_x, ghost_y),
                                        self._target, 
                                        self._matrix, 
                                        prev
                                        )
        self._t = 0
        self.next_tile = (ghost_x + self._direction[0], 
                          ghost_y + self._direction[1])
        self.prev = (ghost_x, ghost_y)

    @abstractmethod
    def determine_target(self):
        ...
        
    def get_target_pacman_dir(self, pacman_rect: tuple, 
                              pacman_dir: tuple,
                              look_ahead: int=4):
        match pacman_dir:
            case "l":
                target = (pacman_rect[0], pacman_rect[1] - look_ahead)
                if target[1] < 0:
                    target = (pacman_rect[0], self.num_cols - look_ahead - 1)
                return target
            
            case "r":
                target = (pacman_rect[0], pacman_rect[1] + look_ahead)
                if target[1] > self.num_cols:
                    target = (pacman_rect[0], 0)
                return target
            case "u":
                return (pacman_rect[0] - look_ahead, pacman_rect[1])
            case "d":
                return (pacman_rect[0] + look_ahead, pacman_rect[1])
            case _:
                return pacman_rect

    def get_random_target(self):
        rand_row = random.randrange(0, self.num_rows)
        rand_col = random.randrange(0, self.num_cols)
        return rand_row, rand_col
    
    def make_ghost_scared(self):
        self._direction = self._direction_prevent[self._direction]
        self.is_scared = True
        self.prepare_movement()

    def check_if_pacman_powered(self):
        if not self._is_released:
            if self.image != self.normal_image:
                self.image = self.normal_image
            return
        if self._game_state.power_event_trigger_time is not None and \
                self.release_time > self._game_state.power_event_trigger_time:
            return
        if self._game_state.is_pacman_powered:
            if self.image != self.blue_image:
                self.image = self.blue_image
                self.make_ghost_scared()
        else:
            if self.image != self.normal_image:
                self.image = self.normal_image
                self.is_scared = False

    def reset_ghost(self):
        self._t = 0
        self._direction = None
        self._target = None
        self._curr_pos = None
        self.prev = None
        self.next_tile = None
        self.release_time = None
        self.is_scared = False
        self.rect.x
        x, y = self._get_coords_from_idx(self._ghost_matrix_pos)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.rect_x = x
        self.rect_y = y
        self._is_released = False
        self._creation_time = pytime.get_ticks()

    def check_collisions(self):
        ghost_rect = Rect(self.rect.x, self.rect.y, 
                          PACMAN[0]//2, PACMAN[1]//2)
        pacman_coords = (self._game_state.pacman_rect[0],
                         self._game_state.pacman_rect[1],
                         self._game_state.pacman_rect[2]//2,
                         self._game_state.pacman_rect[3]//2)
        pacman_rect = Rect(pacman_coords)
        if ghost_rect.colliderect(pacman_rect):
            if self.is_scared:
                self.reset_ghost()  
                self.sounds.play_sound("eat_ghost")
                self._game_state.points += GHOST_POINT
            else:
                self._game_state.is_pacman_dead = True
                self.sounds.play_sound("death")
                wait(1000)

    def update(self, dt):
        self.build_bounding_boxes(self.rect_x, self.rect_y)
        self.check_is_released()
        self._boundary_check()
        self.move_ghost()
        self.check_if_pacman_powered()
        self.check_collisions()

class Blinky(Ghost):
    def determine_target(self):
        mode = self._game_state.ghost_mode
        match mode:
            case "scatter":
                target = GHOST_SCATTER_TARGETS[self.name]
            case "chase":
                pacman_rect = self._game_state.pacman_rect
                target = self._get_idx_from_coords((pacman_rect[0], pacman_rect[1]))
        return target
    
class Pinky(Ghost):
    def calculate_pacman_direction(self):
        pacman_dir = self._game_state.pacman_direction
        pacman_rect = self._game_state.pacman_rect
        pacman_rect = self._get_idx_from_coords((pacman_rect[0], 
                                                        pacman_rect[1]))
        return self.get_target_pacman_dir(pacman_rect, 
                                          pacman_dir,
                                          )

    def determine_target(self):
        mode = self._game_state.ghost_mode
        match mode:
            case "scatter":
                return GHOST_SCATTER_TARGETS[self.name]
            case "chase":
                return self.calculate_pacman_direction()
            
class Inky(Ghost):
    def calculate_inky_target(self):
        pacman_rect = self._game_state.pacman_rect
        pacman_rect = self._get_idx_from_coords((pacman_rect[0], pacman_rect[1]))
        pacman_dir = self._game_state.pacman_direction
        blinky_cell = self._game_state.blinky_matrix_pos
        inky_pacman_target = self.get_target_pacman_dir(pacman_rect,
                                                        pacman_dir,
                                                        2)
        vec_row = inky_pacman_target[0] - blinky_cell[0]
        vec_col = inky_pacman_target[1] - blinky_cell[1]
        target_row = blinky_cell[0] + vec_row * 2
        target_col = blinky_cell[1] + vec_col * 2
        return target_row, target_col  

    def determine_target(self):
        mode = self._game_state.ghost_mode
        match mode:
            case "scatter":
                return GHOST_SCATTER_TARGETS[self.name]
            case "chase":
                return self.calculate_inky_target()
            
class Clyde(Ghost):
    def get_clyde_random_target(self):
        pacman_rect = self._game_state.pacman_rect
        pacman_rect = self._get_idx_from_coords((pacman_rect[0], pacman_rect[1]))
        if not self.curr_pos:
            return pacman_rect
        dis = abs(pacman_rect[0] - self.curr_pos[0]) + abs(pacman_rect[1] - self.curr_pos[1])
        if dis > 8:
            return self.get_random_target()
        return pacman_rect
        
    def determine_target(self):
        mode = self._game_state.ghost_mode
        match mode:
            case "scatter":
                return GHOST_SCATTER_TARGETS[self.name]
            case "chase":
                return self.get_clyde_random_target()

class GhostManager:
    def __init__(self,
                 screen: Surface,
                 game_state: GameState,
                 matrix: list[list[str]],
                 ghost_matrix_pos: tuple[int, int],
                 grid_start_pos: tuple[int, int],
                 ):
        self.screen = screen
        self.game_state = game_state
        self.matrix = matrix
        self.ghost_matrix_pos = ghost_matrix_pos
        self.grid_start_pos = grid_start_pos
        self.ghosts_list = []
        self.load_ghosts()
    
    def load_ghosts(self):
        # ghost_pos = self.ghost_matrix_pos
        adder = 0
        ghosts = [('blinky', Blinky), 
                  ('pinky', Pinky), 
                  ('inky', Inky), 
                  ('clyde', Clyde)]
        for ghost_name, ghost in ghosts:
            ghost_pos = (self.ghost_matrix_pos[0], self.ghost_matrix_pos[1] + adder)
            self.ghosts_list.append(ghost(ghost_name,
                                          ghost_pos,
                                          self.grid_start_pos,
                                          self.matrix,
                                          self.game_state))
            adder += 1
# ===== File: PyPacman-main/main.py =====


# ===== File: PyPacman-main/src/runner.py =====
import sys

import pygame
import json

logger = get_logger(__name__)

class GameRun:
    def __init__(self):
        logger.info("About to initialize pygame")
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Py-Pacman")
        logger.info("pygame initialized")
        self.game_state = GameState()
        logger.info("game state object created")
        self.events = EventHandler(self.screen, self.game_state)
        logger.info("event handler object created")
        self.all_sprites = pygame.sprite.Group()
        self.gui = ScreenManager(self.screen, self.game_state, self.all_sprites)
        logger.info("screen manager object created")

    def initialize_highscore(self):
        with open("levels/stats.json") as fp:
            stats = json.load(fp)
            self.game_state.highscore = stats['highscore']
            self.game_state.mins_played = stats['mins_played']
    
    def create_ghost_mode_event(self):
        CUSTOM_EVENT = pygame.USEREVENT + 1
        pygame.time.set_timer(CUSTOM_EVENT, 
                              self.game_state.mode_change_events * 1000)
        self.game_state.custom_event = CUSTOM_EVENT

    def initialize_sounds(self):
        sound_manager = SoundManager()
        sound_manager.load_sound("dot", "assets/sounds/pacman_chomp.wav", channel=0)
        sound_manager.load_sound("death","assets/sounds/pacman_death.wav", 0.7, 500, 1)
        sound_manager.load_sound("eat_ghost","assets/sounds/pacman_eatghost.wav", 0.6, 100, 2)
        sound_manager.set_background_music("assets/sounds/backgroud.mp3")
        sound_manager.play_background_music()

    def check_highscores(self):
        if self.game_state.points > self.game_state.highscore:
            self.game_state.highscore = self.game_state.points

    def update_highscore(self):
        with open("levels/stats.json", 'w') as fp:
            json.dump({"highscore":self.game_state.highscore,
                       "mins_played": self.game_state.mins_played}, fp, indent=4)
            
    def main(self):
        clock = pygame.time.Clock()
        dt = None
        self.create_ghost_mode_event()
        self.initialize_sounds()
        self.initialize_highscore()
        while self.game_state.running:
            self.game_state.current_time = pygame.time.get_ticks()
            for event in pygame.event.get():
                self.events.handle_events(event)
            self.screen.fill(Colors.BLACK)
            self.gui.draw_screens()
            self.all_sprites.draw(self.screen)
            self.all_sprites.update(dt)
            self.check_highscores()
            pygame.display.flip()
            dt = clock.tick(self.game_state.fps)
            dt /= 100
        self.update_highscore()
        pygame.quit()
        sys.exit()

# ===== File: PyPacman-main/src/__init__.py =====

# ===== File: PyPacman-main/src/game/__init__.py =====

# ===== File: PyPacman-main/src/game/event_management.py =====
from pygame import (K_DOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_SPACE, K_UP, KEYDOWN,
                    QUIT, K_q)
from pygame import USEREVENT
from pygame.time import set_timer

class EventHandler:
    def __init__(self, screen, game_state):
        self._screen = screen
        self._game_screen = game_state

    def pygame_quit(self):
        self._game_screen.running = False

    def key_bindings(self, key):
        if key == K_LEFT:
            self._game_screen.direction = "l"
        elif key == K_RIGHT:
            self._game_screen.direction = "r"
        elif key == K_UP:
            self._game_screen.direction = "u"
        elif key == K_DOWN:
            self._game_screen.direction = "d"

    def handle_events(self, event):
        if event.type == QUIT:
            self.pygame_quit()

        if event.type == KEYDOWN:
            self.key_bindings(event.key)
        
        if event.type == self._game_screen.custom_event:
            curr_mode = self._game_screen.ghost_mode
            if curr_mode == 'scatter':
                self._game_screen.ghost_mode = 'chase'
            elif curr_mode == 'chase':
                self._game_screen.ghost_mode = 'scatter'
            CUSTOM_EVENT = USEREVENT + 1
            set_timer(CUSTOM_EVENT, 
                                self._game_screen.mode_change_events * 1000)
            self._game_screen.custom_event = CUSTOM_EVENT
        
        if event.type == self._game_screen.power_up_event:
            self._game_screen.is_pacman_powered=False


# ===== File: PyPacman-main/src/game/state_management.py =====

class GameState:
    def __init__(self):
        self.__level = 1
        self.__running = True
        self.__fps = 60
        self.__direction = ""
        self.__current_time = None
        self.__pacman_rect = None
        self.__ghost_pos = {}
        self.__is_loaded = False
        self.__is_pacman_powered = False
        self._ghost_mode = 'scatter'
        self._mode_change_events = None
        self.__current_mode_index = 0
        self._custom_event = None
        self._pacman_direction = None
        self._blinky_matrix_pos = None
        self._scared_time = None
        self._power_up_event = None
        self._power_event_trigger_time = None
        self._is_pacman_dead = False
        self._highscore = 0
        self._mins_played = 0
        self._points = -DOT_POINT
        self._level_complete = False

    @property
    def level_complete(self):
        return self._level_complete
    
    @level_complete.setter
    def level_complete(self, val):
        self._level_complete = val

    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, val):
        self._points = val

    @property
    def highscore(self):
        return self._highscore
    
    @highscore.setter
    def highscore(self, val):
        self._highscore = val
    
    @property
    def mins_played(self):
        return self._mins_played
    
    @mins_played.setter
    def mins_played(self, val):
        self._mins_played = val

    @property
    def is_pacman_dead(self):
        return self._is_pacman_dead
    
    @is_pacman_dead.setter
    def is_pacman_dead(self, val):
        self._is_pacman_dead = val

    @property
    def power_event_trigger_time(self):
        return self._power_event_trigger_time
    
    @power_event_trigger_time.setter
    def power_event_trigger_time(self, val):
        self._power_event_trigger_time = val

    @property
    def power_up_event(self):
        return self._power_up_event
    
    @power_up_event.setter
    def power_up_event(self, val):
        self._power_up_event = val

    @property
    def scared_time(self):
        return self._scared_time
    
    @scared_time.setter
    def scared_time(self, val):
        self._scared_time = val

    @property
    def blinky_matrix_pos(self):
        return self._blinky_matrix_pos
    
    @blinky_matrix_pos.setter
    def blinky_matrix_pos(self, val):
        self._blinky_matrix_pos = val

    @property
    def pacman_direction(self):
        return self._pacman_direction
    
    @pacman_direction.setter
    def pacman_direction(self, val):
        self._pacman_direction = val

    @property
    def custom_event(self):
        return self._custom_event
    
    @custom_event.setter
    def custom_event(self, val):
        self._custom_event = val

    @property
    def mode_change_events(self):
        if self.__current_mode_index >= len(self._mode_change_events):
            curr_event = self._mode_change_events[-1]
        else:
            curr_event = self._mode_change_events[self.__current_mode_index]
            self.__current_mode_index += 1
        return curr_event

    @mode_change_events.setter
    def mode_change_events(self, val):
        self._mode_change_events = val

    @property
    def ghost_mode(self):
        return self._ghost_mode
    
    @ghost_mode.setter
    def ghost_mode(self, value):
        if value not in ['scatter', 'chase', 'scared']:
            raise ValueError("Only scatter, scared or chase modes are available")
        self._ghost_mode = value

    @property
    def is_pacman_powered(self):
        return self.__is_pacman_powered
    
    @is_pacman_powered.setter
    def is_pacman_powered(self, val):
        self.__is_pacman_powered = val
        
    @property
    def is_loaded(self):
        return self.__is_loaded

    def get_ghost_pos(self, name):
        return self.__ghost_pos.get(name)
    
    def set_ghost_pos(self, name, val):
        self.__ghost_pos[name] = val

    @property
    def pacman_rect(self):
        return self.__pacman_rect
    
    @pacman_rect.setter
    def pacman_rect(self, rect):
        self.__pacman_rect = rect
    
    @property
    def current_time(self):
        return self.__current_time
    
    @current_time.setter
    def current_time(self, val):
        self.__current_time = val

    @property
    def direction(self):
        return self.__direction

    @direction.setter
    def direction(self, value):
        if value not in ["r", "l", "u", "d", ""]:
            raise ValueError("Unknown direction")
        self.__direction = value

    @property
    def level(self):
        return self.__level

    @level.setter
    def level(self, value):
        self.__level = value

    @property
    def running(self):
        return self.__running

    @running.setter
    def running(self, value):
        self.__running = value

    @property
    def fps(self):
        return self.__fps

    @fps.setter
    def fps(self, value):
        self.__fps = value

# ===== File: PyPacman-main/src/gui/__init__.py =====

# ===== File: PyPacman-main/src/gui/loading_screen.py =====
from pygame import image, transform

class LoadingScreen:
    def __init__(self, screen):
        self.screen = screen
        self.loading_image = image.load(loading_screen_gif)  # Load the image
        self.loading_image = transform.scale(self.loading_image, (192, 192))

    def draw_loading(self):
        self.screen.blit(self.loading_image, (500, 500))
# ===== File: PyPacman-main/src/gui/pacman_grid.py =====
import json
logger = get_logger(__name__)

class PacmanGrid:
    def __init__(self, screen, game_state):
        logger.info("initializing pacman grid")
        self.function_mapper = {
            "void": self.draw_void,
            "wall": self.draw_wall,
            "dot": self.draw_dot,
            "spoint": self.draw_special_point,
            "power": self.draw_power,
            "null": self.draw_void,
            "elec": self.draw_elec,
        }
        self._screen = screen
        self._game_state = game_state
        self._level_number = self._game_state.level
        self.load_level(self._level_number)
        logger.info("level loaded")
        self.pacman = Pacman(
            self._screen,
            self._game_state,
            self._matrix,
            self._pacman_pos,
            (self.start_x, self.start_y)
        )
        self.ghost = GhostManager(
            self._screen,
            self._game_state,
            self._matrix,
            self.ghost_den,
            (self.start_x, self.start_y)
        )
        logger.info("pacman created")
        
    def get_json(self, path):
        with open(path) as fp:
            payload = json.load(fp)
        return payload

    def load_level(self, level_number):
        level_path = f"levels/level{level_number}.json"
        level_json = self.get_json(level_path)
        num_rows = level_json["num_rows"]
        num_cols = level_json["num_cols"]
        self.ghost_den = level_json['ghost_den']
        self._matrix = level_json["matrix"]
        self._pacman_pos = level_json["pacman_start"]
        self.elec_pos = level_json['elec']
        self.mode_change_times = level_json['scatter_times']
        self.power_up_time = level_json['power_up_time']
        self._game_state.scared_time = self.power_up_time
        self._game_state.mode_change_events = self.mode_change_times
        self.start_x, self.start_y = place_elements_offset(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            CELL_SIZE[0] * num_cols,
            CELL_SIZE[0] * num_rows,
            0.5,
            0.5,
        )
        self._coord_matrix = precompute_matrix_coords(
            self.start_x, self.start_y, CELL_SIZE[0], num_rows, num_cols
        )
        self.num_rows = num_rows
        self.num_cols = num_cols

    def draw_void(self, **kwargs): ...

    def draw_wall(self, **kwargs):
        draw_rect(
            kwargs["x"],
            kwargs["y"],
            kwargs["w"],
            kwargs["h"],
            self._screen,
            Colors.WALL_BLUE,
        )

    def draw_dot(self, **kwargs):
        dot_x = kwargs["x"] + kwargs["w"]
        dot_y = kwargs["y"] + kwargs["h"]
        draw_rect(dot_x, dot_y, 5, 5, self._screen, Colors.WHITE)

    def draw_special_point(self): ...

    def draw_power(self, **kwargs):
        circle_x = kwargs["x"] + kwargs["w"]
        circle_y = kwargs["y"] + kwargs["h"]
        draw_circle(circle_x, circle_y, 7, self._screen, Colors.YELLOW)

    def draw_elec(self, **kwargs):
        draw_rect(kwargs["x"], kwargs["y"], kwargs["w"], 1, self._screen, Colors.RED)

    def draw_level(self):
        curr_x, curr_y = self.start_x, self.start_y
        for _, row in enumerate(self._matrix):
            for _, col in enumerate(row):
                draw_func = self.function_mapper[col]
                draw_func(x=curr_x, y=curr_y, w=CELL_SIZE[0], h=CELL_SIZE[0])
                curr_x += CELL_SIZE[0]
            curr_x = self.start_x
            curr_y += CELL_SIZE[0]

    def reset_stage(self):
        self.pacman = Pacman(
            self._screen,
            self._game_state,
            self._matrix,
            self._pacman_pos,
            (self.start_x, self.start_y)
        )
        self.ghost = GhostManager(
            self._screen,
            self._game_state,
            self._matrix,
            self.ghost_den,
            (self.start_x, self.start_y)
        )
        
    def draw_outliners(self):
        draw_debug_rects(
            self.start_x, self.start_y, 128, 140, 5, Colors.GREEN, self._screen
        )
        draw_debug_rects(
            self.start_x,
            self.start_y,
            self.num_rows,
            self.num_cols,
            CELL_SIZE[0],
            Colors.BLUE,
            self._screen,
        )

# ===== File: PyPacman-main/src/gui/score_screen.py =====
from pygame.surface import Surface
from pygame import font


class ScoreScreen:
    def __init__(self,
                 screen: Surface,
                 game_state: GameState):
        self._screen = screen
        self._game_state = game_state
        self.start_x, self.start_y = place_elements_offset(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            200, 
            200,
            0.25,
            0.05
        )
        font.init()
        self.font = font.Font(None, 36)

    def draw_scores(self):
        score_text = "SCORE: " + str(self._game_state.points)
        score_surface = self.font.render(score_text, True, Colors.WHITE)
        self._screen.blit(score_surface, (self.start_x, self.start_y))

        highscore_text = "HIGHSCORE: "+str(self._game_state.highscore)
        hs_surface = self.font.render(highscore_text, True, Colors.WHITE)
        self._screen.blit(hs_surface, (self.start_x + 300, self.start_y))
        
# ===== File: PyPacman-main/src/gui/screen_management.py =====

from pygame.time import wait

logger = get_logger(__name__)

class ScreenManager:
    def __init__(self, screen, game_state, all_sprites):
        logger.info("screen manager initializing")
        self._screen = screen
        self._game_state = game_state
        self.all_sprites = all_sprites
        self.loading_screen = LoadingScreen(self._screen)
        self.pacman = PacmanGrid(screen, game_state)
        self.score_screen = ScoreScreen(self._screen, self._game_state)
        logger.info("pacman grid created")
        self.all_sprites.add(self.pacman.pacman)
        for ghost in self.pacman.ghost.ghosts_list:
            self.all_sprites.add(ghost)

    def pacman_dead_reset(self):
        if self._game_state.is_pacman_dead:
            self._game_state.is_pacman_dead = False
            self._game_state.direction = ""
            self._game_state.pacman_direction = None
            self.all_sprites.empty()
            self.pacman.reset_stage()
            self.all_sprites.add(self.pacman.pacman)
            for ghost in self.pacman.ghost.ghosts_list:
                self.all_sprites.add(ghost)
    
    def check_level_complete(self):
        if self._game_state.level_complete:
            wait(2000)
            self.all_sprites.empty()
            self.pacman = PacmanGrid(self._screen, self._game_state)
            self.score_screen = ScoreScreen(self._screen, self._game_state)
            logger.info("pacman grid created")
            self.all_sprites.add(self.pacman.pacman)
            for ghost in self.pacman.ghost.ghosts_list:
                self.all_sprites.add(ghost)
            self._game_state.level_complete = False

    def draw_screens(self):
        self.pacman.draw_level()
        self.pacman_dead_reset()
        self.score_screen.draw_scores()
        self.check_level_complete()

# ===== File: PyPacman-main/src/log_handle.py =====
import logging

def get_logger(name: str, level: int = logging.DEBUG):
    """
    Returns a preconfigured logger instance.
    
    Parameters:
        name (str): Name of the logger, typically __name__ of the module.
        level (int): Logging level. Defaults to DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if logger already has handlers to avoid duplicate logs
    if not logger.hasHandlers():
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Define a standard format for all logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger

# ===== File: PyPacman-main/src/sounds.py =====
import pygame


class SoundManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SoundManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._sounds = {}
            self._channels = {}
            self._background_music = None
            pygame.mixer.pre_init()
            pygame.mixer.set_num_channels(64)
            # pygame.mixer.init()
    
    def load_sound(self, name, filepath, 
                   volumne=0.5, 
                   freq=200,
                   channel=0):
        """Loads a sound effect and assigns it a name."""
        self._sounds[name] = {"sound": pygame.mixer.Sound(filepath),
                              "freq": freq,
                              'last_played': 0}
        self._sounds[name]['sound'].set_volume(volumne)
        self._channels[name] = pygame.mixer.Channel(channel)

    def play_sound(self, name):
        """Plays a specific sound effect."""
        if name in self._sounds:
            # if not pygame.mixer.get_busy():
                now = pygame.time.get_ticks()
                freq = self._sounds[name]['freq']
                last_played = self._sounds[name]['last_played']
                if now - last_played > freq: 
                    self._channels[name].play(self._sounds[name]['sound'])
                    self._sounds[name]['last_played'] = now
        else:
            print(f"Sound '{name}' not found!")

    def set_background_music(self, filepath):
        """Loads and sets the background music."""
        self._background_music = filepath
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.set_volume(0.2)  # Adjust the volume

    def play_background_music(self, loops=-1, start=0.0, fade_ms=0):
        """Starts playing the background music."""
        if self._background_music:
            pygame.mixer.music.play(loops=loops, start=start, fade_ms=fade_ms)
        else:
            print("Background music not set!")

    def stop_background_music(self):
        """Stops the background music."""
        pygame.mixer.music.stop()

    def stop_all_sounds(self):
        """Stops all currently playing sounds."""
        pygame.mixer.stop()

# ===== File: PyPacman-main/src/sprites/pacman.py =====
from math import ceil

from pygame import image, transform
from pygame.sprite import Sprite
from pygame import Surface, USEREVENT
from pygame.time import set_timer, get_ticks
logger = get_logger(__name__)

class Pacman(Sprite):
    def __init__(self, 
                 screen: Surface, 
                 game_state: GameState, 
                 matrix: list[list[str]],
                 pacman_pos: tuple,
                 start_pos: tuple):
        super().__init__()
        self.screen = screen
        self.game_state = game_state
        self.pacman_pos = pacman_pos
        self.matrix = matrix
        self.start_pos = start_pos
        self.load_all_frames()
        self.calculate_pacman_coords()
        self.load_image()
        self.calculate_tiny_matrix()
        self.calculate_coord_matrix()
        self.frame_delay = 5
        self.sound = SoundManager()
        self.collectibles = self.count_dots_powers()

    def count_dots_powers(self):
        collectibles = 0
        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[0])):
                if col + 1 >= len(self.matrix[0]):
                    continue
                if self.matrix[row][col] in ['dot', 'power'] and \
                        self.matrix[row+1][col] not in ['wall', 'elec', 'null']:
                    collectibles += 1
        # logger.info("total_collectibles: %s",collectibles)
        return collectibles

    def load_image(self):
        self.image = self.frames[self.curr_frame_idx]
        self.rect_x = self.pacman_x_coord
        self.rect_y = self.pacman_y_coord
        self.rect = self.image.get_rect(topleft=(self.pacman_x_coord,
                                                 self.pacman_y_coord))
    
    def build_bounding_boxes(self, x: int | float, y: int | float):
        self.rect.x = x + (CELL_SIZE[0] * 2 - self.rect.width) // 2
        self.rect.y = y + (CELL_SIZE[1] * 2 - self.rect.height) // 2
        
    def frame_update(self):
        self.frame_delay -= 1
        if self.frame_delay <= 0:
            self.frame_delay = 5
            self.curr_frame_idx = (self.curr_frame_idx + 1) % len(self.frames)
            self.image = self.frames[self.curr_frame_idx]

    def frame_direction_update(self):
        if self.move_direction != "":
            self.frames = self.direction_mapper[self.move_direction]

    def calculate_pacman_coords(self):
        x, y = get_coords_from_idx(
            self.pacman_pos,
            self.start_pos[0],
            self.start_pos[1],
            CELL_SIZE[0],
            CELL_SIZE[1],
            len(self.matrix),
            len(self.matrix[0])
        )
        self.pacman_x_coord = x
        self.pacman_y_coord = y
    
    def load_all_frames(self):
        def frame_helper(direction):
            width, height = PACMAN
            return [
                transform.scale(image.load(path).convert_alpha(), (width, height))
                for path in PACMAN_PATHS[direction]
            ]
        self.curr_frame_idx = 0
        self.left_frames = frame_helper("left")
        self.right_frames = frame_helper("right")
        self.down_frames = frame_helper("down")
        self.up_frames = frame_helper("up")
        self.direction_mapper = {
            "l": self.left_frames,
            "r": self.right_frames,
            "u": self.up_frames,
            "d": self.down_frames,
        }
        self.frames = self.right_frames
        self.move_direction = self.game_state.direction

    def calculate_tiny_matrix(self):
        self.tiny_matrix = get_tiny_matrix(self.matrix,
                                           CELL_SIZE[0],
                                           PACMAN_SPEED)
        self.subdiv = CELL_SIZE[0] // PACMAN_SPEED
        self.tiny_start_x = self.pacman_pos[0] * self.subdiv
        self.tiny_start_y = self.pacman_pos[1] * self.subdiv

    def calculate_coord_matrix(self):
        self.coord_matrix = precompute_matrix_coords(*self.start_pos,
                                                     PACMAN_SPEED,
                                                     len(self.tiny_matrix),
                                                     len(self.tiny_matrix[0]))

    def edges_helper_vertical(self, row: int, 
                              col: int, 
                              additive: int):
        for r in range(self.subdiv * 2):
            if self.tiny_matrix[row + r][col + additive] == "wall":
                return False
        return True

    def edge_helper_horizontal(self, row: int, 
                               col: int, 
                               additive: int):
        for c in range(self.subdiv * 2):
            if self.tiny_matrix[row + additive][col + c] == "wall":
                return False
        return True

    def boundary_check(self):
        if (self.tiny_start_y + self.subdiv * 2) >= len(self.tiny_matrix[0]) - 1:
            self.tiny_start_y = 0
            self.rect_x = self.coord_matrix[self.tiny_start_x][0][0]

        elif (self.tiny_start_y - 1) < 0:
            self.tiny_start_y = len(self.tiny_matrix[0]) - (self.subdiv * 3)
            self.rect_x = self.coord_matrix[self.tiny_start_x][-self.subdiv*2 - 4][0]

    def create_power_up_event(self):
        CUSTOM_EVENT = USEREVENT + 2
        set_timer(CUSTOM_EVENT, 
                self.game_state.scared_time)
        self.game_state.power_up_event = CUSTOM_EVENT
        self.game_state.is_pacman_powered = True
        self.game_state.power_event_trigger_time = get_ticks()

    def eat_dots(self):
        r, c = get_idx_from_coords(
            self.rect.x, self.rect.y, *self.start_pos, CELL_SIZE[0]
        )
        match self.matrix[r][c]:
            case "dot":
                self.matrix[r][c] = "void"
                self.sound.play_sound("dot")
                self.collectibles -= 1
                self.game_state.points += DOT_POINT
            case "power":
                self.matrix[r][c] = "void"
                self.create_power_up_event()
                self.sound.play_sound("dot")
                self.collectibles -= 1
                self.game_state.points += POWER_POINT
                
    def movement_bind(self):
        match self.game_state.direction:
            case 'l':
                if self.edges_helper_vertical(self.tiny_start_x, self.tiny_start_y, -1):
                    self.move_direction = "l"
                    self.game_state.pacman_direction = 'l'
            
            case 'r':
                if self.edges_helper_vertical(
                    self.tiny_start_x, self.tiny_start_y, self.subdiv * 2
                ):
                    self.move_direction = "r"
                    self.game_state.pacman_direction = 'r'
            
            case 'u':
                if self.edge_helper_horizontal(self.tiny_start_x, self.tiny_start_y, -1):
                    self.move_direction = "u"
                    self.game_state.pacman_direction = 'u'
            
            case 'd':
                if self.edge_helper_horizontal(
                    self.tiny_start_x, self.tiny_start_y, self.subdiv * 2
                ):
                    self.move_direction = "d" 
                    self.game_state.pacman_direction = 'd'
 
    def move_pacman(self, dt: float):
        match self.move_direction:
            case "l":
                if self.edges_helper_vertical(self.tiny_start_x, self.tiny_start_y, -1):
                    self.rect_x -= PACMAN_SPEED
                    self.tiny_start_y -= 1
            case "r":
                if self.edges_helper_vertical(
                self.tiny_start_x, self.tiny_start_y, self.subdiv * 2
            ):
                    self.rect_x += PACMAN_SPEED
                    self.tiny_start_y += 1

            case "u":
                if self.edge_helper_horizontal(self.tiny_start_x, self.tiny_start_y, -1):
                    self.rect_y -= PACMAN_SPEED
                    self.tiny_start_x -= 1
            
            case "d":
                if self.edge_helper_horizontal(
                self.tiny_start_x, self.tiny_start_y, self.subdiv * 2
            ):
                    self.rect_y += PACMAN_SPEED
                    self.tiny_start_x += 1

        self.game_state.pacman_rect = (self.rect_x, self.rect_y, 
                                       CELL_SIZE[0]*2, CELL_SIZE[0]*2)

    def update(self, dt: float):
        self.frame_update()
        self.build_bounding_boxes(self.rect_x, self.rect_y)
        self.movement_bind()
        self.move_pacman(dt)
        self.boundary_check()
        self.eat_dots()
        self.frame_direction_update()
        if self.collectibles == 0:
            self.game_state.level_complete = True
# ===== File: PyPacman-main/src/utils/__init__.py =====


# ===== Consolidated __main__ blocks =====
if __name__ == '__main__':
    # from PyPacman-main/main.py
    gr = GameRun()
    gr.main()
