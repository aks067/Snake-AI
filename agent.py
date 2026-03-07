import random
from collections import deque

import torch

from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001
GAMMA = 0.9
HIDDEN_SIZE = 256
INPUT_SIZE = 11
OUTPUT_SIZE = 3

EPSILON_START = 80

class Agent:
	def __init__(self, model_path: str = None):
		self.n_games = 0
		self.epsilon = 0
		self.memory = deque(maxlen=MAX_MEMORY)

		self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
		self.trainer = QTrainer(self.model, learningRate=LR, gamma=GAMMA)
		if model_path:
			try:
				self.model.load(model_path)
			except FileNotFoundError:
				print(f'[Agent] No saved model at {model_path}, starting fresh')

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def train_short_memory(self, state, action, reward, next_state, done):
		self.trainer.train_step(state, action, reward, next_state, done)

	def train_long_memory(self):
		if len(self.memory) < BATCH_SIZE:
			sample = list(self.memory)
		else:
			sample = random.sample(self.memory, BATCH_SIZE)

		states, actions, rewards, next_states, dones = zip(*sample)
		self.trainer.train_step(list(states), list(actions), list(rewards), list(next_states), list(dones))


	def get_action(self, state) -> list[int]:
		self.epsilon = EPSILON_START - self.n_games
		action = [0 ,0 ,0]

		if random.randint(0, 200) < self.epsilon:
			action[random.randint(0, 2)] = 1
		else:
			state_tensor = torch.tensor(state, dtype=torch.float)
			with torch.no_grad():
				prediction = self.model(state_tensor)
			action[torch.argmax(prediction).item()] = 1
		return action