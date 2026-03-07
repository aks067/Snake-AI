import pygame
import time
import random
from enum import Enum
from collections import namedtuple

BLOCK = 20
SPEED = 15

WIN_X = 480
WIN_Y = 480

WHITE	= (255, 255, 255)
BLACK	= (0, 0, 0)
RED		= (193, 41, 46)
GREEN1	= (122, 199, 79)
GREEN2	= (80, 160, 40)
BLUE	= (18, 69, 89)
GRAY	= (40, 40, 40)

class Direction(Enum):
	RIGHT	= 0
	DOWN	= 1
	LEFT	= 2
	UP		= 3

Point = namedtuple('Point', ['x', 'y'])

class Snake_Game:
	def __init__(self, render: bool = True, x: int = WIN_X, y: int = WIN_Y):
		self.x = x
		self.y = y
		self.render = render

		if self.render:
			pygame.init()
			self.display = pygame.display.set_mode((x, y))
			pygame.display.set_caption('Snake AI')
			self.clock = pygame.time.Clock()
		self.reset()

	def reset(self):
		self.direction = Direction.RIGHT
		head = Point(self.x // 2, self.y // 2)
		self.snake = [
			head,
			Point(head.x - BLOCK, head.y),
			Point(head.x - 2 * BLOCK, head.y)
		]
		self.score = 0
		self.frame_iter = 0
		self._place_food()
		return self._get_state()
	
	def step(self, action):
		self.frame_iter += 1
		
		if self.render:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					raise SystemExit
		self._move(action)
		self.snake.insert(0, self.head)

		reward = 0
		done = False
		
		if self._is_collision() or self.frame_iter > 100 * len(self.snake):
			done = True
			reward = -10
			if self.render:
				self._draw()
			return self._get_state(), reward, done, self.score
		
		if self.head == self.food:
			self.score += 1
			reward = 10
			self._place_food()
		else:
			self.snake.pop()

		if self.render:
			self._draw()
			self.clock.tick(SPEED)
		
		return self._get_state(), reward, done, self.score
	
	def _get_state(self):
		head = self.snake[0]
		dir = self.direction

		point_right = Point(head.x + BLOCK, head.y)
		point_left = Point(head.x - BLOCK, head.y)
		point_up = Point(head.x, head.y - BLOCK)
		point_down = Point(head.x, head.y + BLOCK)

		dir_right = dir == Direction.RIGHT
		dir_left = dir == Direction.LEFT
		dir_up = dir == Direction.UP
		dir_down = dir == Direction.DOWN

		state = [
			(dir_right and self._is_collision(point_right)) or
			(dir_left and self._is_collision(point_left)) or
			(dir_up and self._is_collision(point_up)) or
			(dir_down and self._is_collision(point_down)),

			(dir_right and self._is_collision(point_down)) or
			(dir_left and self._is_collision(point_up)) or
			(dir_up and self._is_collision(point_right)) or
			(dir_down and self._is_collision(point_left)),

			(dir_right and self._is_collision(point_up)) or
			(dir_left and self._is_collision(point_down)) or
			(dir_up and self._is_collision(point_left)) or
			(dir_down and self._is_collision(point_right)),

			dir_left, dir_right, dir_up, dir_down,

			self.food.x < head.x,
			self.food.x > head.x,
			self.food.y < head.y,
			self.food.y > head.y,
		]
		return [int(s) for s in state]
	
	def _place_food(self):
		cols = self.x // BLOCK
		rows = self.y // BLOCK
		while True:
			x = random.randint(0, cols - 1) * BLOCK
			y = random.randint(0, rows - 1) * BLOCK
			self.food = Point(x, y)
			if self.food not in self.snake:
				break

	def _is_collision(self, point=None):
		if point is None:
			point = self.snake[0]

		if point.x < 0 or point.x >= self.x or point.y < 0 or point.y >= self.y:
			return True
		if point in self.snake[1:]:
			return True
		return False
	
	def _move(self, action):
		clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		index = clock_wise.index(self.direction)

		if action == [1, 0 ,0]: #tout droit
			new_dir = clock_wise[index]
		elif action == [0,1, 0]: #droite
			new_dir = clock_wise[(index + 1) % 4]
		else: #gauche
			new_dir = clock_wise[(index - 1) % 4]

		self.direction = new_dir
		self.head = self.snake[0]

		if   self.direction == Direction.RIGHT: self.head = Point(self.head.x + BLOCK, self.head.y)
		elif self.direction == Direction.LEFT:  self.head = Point(self.head.x - BLOCK, self.head.y)
		elif self.direction == Direction.DOWN:  self.head = Point(self.head.x, self.head.y + BLOCK)
		elif self.direction == Direction.UP:    self.head = Point(self.head.x, self.head.y - BLOCK)

	def _draw(self):
		self.display.fill(GRAY)

		for x in range(0, self.x + 1, BLOCK):
			pygame.draw.line(self.display, (60, 60, 60), (x, 0), (x, self.y))
		for y in range(0, self.y + 1, BLOCK):
			pygame.draw.line(self.display, (60, 60, 60), (0, y), (self.x, y))

		for i, point in enumerate(self.snake):
			color = GREEN1 if i == 0 else GREEN2
			pygame.draw.rect(self.display, color, pygame.Rect(point.x, point.y, BLOCK - 2 , BLOCK - 2))

		pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x + 2, self.food.y + 2, BLOCK - 4, BLOCK - 4))

		pygame.display.set_caption(f'Snake AI  | Score: {self.score}')
		pygame.display.flip()
