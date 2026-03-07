import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
	def __init__(self, input_size: int, hidden_size: int, output_size: int):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, output_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)
	
	def save(self, path: str = 'model.pth'):
		os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
		torch.save(self.state_dict(), path)
		print(f'[Model] Saved -> {path}')

	def load(self, path: str = 'model.pth'):
		self.load_state_dict(torch.load(path, weights_only=True))
		self.eval()
		print(f'[Model] Loaded <- {path}')

class QTrainer:
	def __init__(self, model: LinearQNet, learningRate: float, gamma: float):
		self.model = model
		self.gamma = gamma
		self.optimizer = optim.Adam(model.parameters(), lr=learningRate)
		self.criterion = nn.MSELoss()

	def train_step(self, state, action, reward, next_state, done):
		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)

		if state.dim() == 1:
			state = state.unsqueeze(0)
			next_state = next_state.unsqueeze(0)
			action = action.unsqueeze(0)
			reward = reward.unsqueeze(0)
			done = (done,)

		prediction = self.model(state)

		target = prediction.clone()
		for i in range(len(done)):
			Q_new = reward[i]
			if not done[i]:
				Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
			target[i][torch.argmax(action[i]).item()] = Q_new
		self.optimizer.zero_grad()
		loss = self.criterion(target, prediction)
		loss.backward()
		self.optimizer.step()

	