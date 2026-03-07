import argparse
import sys

from agent import Agent
from snake import Snake_Game

MODEL_PATH = 'model.pth'

def train(render: bool = True, resume: bool = False):
	scores = []
	best_score = 0
	total_score = 0

	agent = Agent(model_path=MODEL_PATH if resume else None)
	game = Snake_Game(render=render)

	print('Training started. Press Ctrl+C to stop')
	print(f'{"Game":>6} {"Score":>6} {"Best":>6} {"Avg":>7}')
	print('-' * 32)

	try:
		while True:
			state_old = game.reset()

			done = False
			while not done:
				action = agent.get_action(state_old)

				state_new, reward, done, score = game.step(action)

				agent.train_short_memory(state_old, action, reward, state_new, done)
				agent.remember(state_old, action, reward, state_new, done)

				state_old = state_new

			agent.n_games += 1
			agent.train_long_memory()

			if score > best_score:
				best_score = score
				agent.model.save(MODEL_PATH)

			scores.append(score)
			total_score += score
			avg = total_score / agent.n_games

			print(f'{agent.n_games:>6} {score:>6} {best_score:>6} {avg:>7.2f}')
	except KeyboardInterrupt:
		print('\nTraining stopped')
		agent.model.save(MODEL_PATH)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train Snake AI')
	parser.add_argument('--no-render', action='store_true', help='Disable pygame window (faster training)')
	parser.add_argument('--resume', action='store_true', help='Resume from saved model.pth')

	args = parser.parse_args()

	train(render=not args.no_render, resume=args.resume)