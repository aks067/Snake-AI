# 🐍 Snake AI — Deep Q-Network (DQN) with PyTorch

A Snake game where an AI agent learns to play using **Deep Reinforcement Learning** (DQN).  
Built with **Python**, **PyTorch** and **Pygame**.

---


![](https://github.com/aks067/Snake-AI/blob/main/Training%20AI.gif)

---


## Architecture

```
SnakeAI/
├── snake.py      # Snake environment (logic + pygame rendering)
├── model.py     # Neural network (LinearQNet) + QTrainer
├── agent.py     # DQN agent (ε-greedy, replay memory)
├── train.py     # Training loop
└── model.pth    # Saved model weights (generated after training)
```

## How it works

The agent observes an **11-value state vector** at each step:

| # | Feature |
|---|---------|
| 0 | Danger straight |
| 1 | Danger right (relative) |
| 2 | Danger left (relative) |
| 3-6 | Current direction (L/R/U/D) |
| 7-10 | Food position relative to head (L/R/U/D) |

It picks one of **3 actions**: go straight, turn right, turn left.

**Rewards:**
- `+10` — ate food
- `-10` — died (wall or self-collision)

The network is a simple **3-layer MLP** trained with the **Bellman equation**:

$$Q(s, a) = r + \gamma \cdot \max_{a'} Q(s', a')$$

---

## Quick start

### 1. Install dependencies
```bash
pip install torch pygame
```

### 2. Train the AI (with visual)
```bash
python train.py
```

### 3. Train headless (faster)
```bash
python train.py --no-render
```

### 4. Resume training from a saved model
```bash
python train.py --resume
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Discount factor γ | 0.9 |
| Replay memory size | 100 000 |
| Batch size | 1 000 |
| Hidden layer size | 256 |
| Exploration games (ε) | 80 |

---

## Results

The agent typically reaches scores of **30–50** after ~200 games, and improves further with more training.
