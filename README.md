# Game Strategy Optimization using Q-Learning

This project implements a Q-Learning agent that learns to play Tic-Tac-Toe optimally. The implementation includes a complete state space representation, Q-learning setup, and performance evaluation tools.

## Project Structure

- `tictactoe_env.py`: Implements the Tic-Tac-Toe environment with state management and game rules
- `q_learning_agent.py`: Contains the Q-Learning agent implementation
- `train.py`: Training script with visualization and evaluation tools
- `requirements.txt`: Project dependencies

## Features

1. **State Space Representation**
   - Complete board state tracking
   - Valid action management
   - Win/draw detection

2. **Q-Learning Implementation**
   - Epsilon-greedy exploration strategy
   - Dynamic learning rate
   - Reward shaping for wins/draws/invalid moves

3. **Performance Evaluation**
   - Win rate tracking
   - Reward history visualization
   - Training progress plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To train the agent and see its performance:

```bash
python train.py
```

This will:
1. Train the agent for 10,000 episodes
2. Generate training visualization plots
3. Show an example game with the trained agent

## Results Analysis

The training process generates:
- Win rate over time plot
- Reward history plot
- Final win rate evaluation
- Example game demonstration

## Performance Metrics

The agent is evaluated on:
1. Win rate against random play
2. Learning convergence speed
3. Final policy quality
4. Adaptation to opponent strategies
