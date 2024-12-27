import numpy as np
from tictactoe_env import TicTacToeEnv
from q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import random

def display_game_step(env, state, action, q_values, step):
    print(f"\nStep {step}:")
    print("Current Board:")
    env.render()
    if action:
        print(f"Action taken: {action}")
        print("Q-values for available actions:")
        for act, q_val in q_values.items():
            print(f"Action {act}: {q_val:.3f}")
    print("-" * 30)

def demonstrate_game(agent, game_number):
    env = TicTacToeEnv()
    state = env.reset()
    total_reward = 0
    step = 0
    
    print(f"\n{'=' * 60}")
    print(f"{'First' if game_number == 1 else 'Final'} Game Demonstration")
    print(f"{'=' * 60}")
    print("\nInitial Board:")
    env.render()
    
    while True:
        valid_actions = env.get_valid_actions()
        action = agent.get_action(state, valid_actions)
        
        if action is None:
            break
        
        # Display Q-values and decision process
        print(f"\nStep {step + 1}:")
        print("\nAvailable Actions and their Q-values:")
        q_values = {a: agent.q_table[state][a] for a in valid_actions}
        for act, q_val in q_values.items():
            print(f"Action {act}: {q_val:.3f}")
        
        print(f"\nChosen Action: {action}")
        if agent.epsilon > random.random():
            print("(Selected randomly due to exploration)")
        else:
            print("(Selected based on highest Q-value)")
        
        next_state, reward, done = env.make_move(action)
        print("\nResulting Board:")
        env.render()
        
        total_reward += reward
        state = next_state
        step += 1
        
        if done:
            print("\nGame Over!")
            print(f"Total Steps: {step}")
            print(f"Total Reward: {total_reward}")
            if reward == 1:
                print("Result: Agent Won! ")
            elif reward == 0.5:
                print("Result: Draw! ")
            else:
                print("Result: Agent Lost! ")
            print(f"{'=' * 60}\n")
            break
    
    return total_reward

def train_agent(episodes=10000):
    env = TicTacToeEnv()
    agent = QLearningAgent()
    
    # Training metrics
    win_rates = []
    rewards_history = []
    evaluation_interval = 100
    
    # Store agents for demonstration
    first_game_agent = None
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        
        # Save agent state after first game
        if episode == 0:
            first_game_agent = deepcopy(agent)
        
        while True:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)
            
            if action is None:
                break
                
            next_state, reward, done = env.make_move(action)
            next_valid_actions = env.get_valid_actions()
            
            # Update Q-values
            agent.learn(state, action, reward, next_state, next_valid_actions)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
        agent.decay_epsilon()
        
        # Evaluate agent periodically
        if (episode + 1) % evaluation_interval == 0:
            win_rate = evaluate_agent(agent)
            win_rates.append(win_rate)
    
    return agent, first_game_agent, win_rates, rewards_history

def evaluate_agent(agent, n_games=100):
    env = TicTacToeEnv()
    wins = 0
    
    for _ in range(n_games):
        state = env.reset()
        while True:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)
            
            if action is None:
                break
                
            state, reward, done = env.make_move(action)
            
            if done:
                if reward == 1:  # Win
                    wins += 1
                break
                
    return wins / n_games

def plot_training_results(win_rates, rewards_history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(win_rates)
    plt.title('Win Rate over Training')
    plt.xlabel('Evaluation (every 100 episodes)')
    plt.ylabel('Win Rate')
    
    plt.subplot(1, 2, 2)
    plt.plot(rewards_history)
    plt.title('Rewards over Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    # Train the agent
    print("Training agent...")
    final_agent, first_game_agent, win_rates, rewards_history = train_agent()
    
    # Plot results
    plot_training_results(win_rates, rewards_history)
    
    # Show example games
    print("\nDemonstrating first game (untrained agent):")
    demonstrate_game(first_game_agent, 1)
    
    print("\nDemonstrating final game (trained agent):")
    demonstrate_game(final_agent, 2)
    
    # Final evaluation
    final_win_rate = evaluate_agent(final_agent, n_games=1000)
    print(f"\nFinal Win Rate: {final_win_rate:.2%}")
    
    # Save example game
    print("\nExample game:")
    env = TicTacToeEnv()
    state = env.reset()
    env.render()
    
    while True:
        valid_actions = env.get_valid_actions()
        action = final_agent.get_action(state, valid_actions)
        
        if action is None:
            break
            
        state, reward, done = env.make_move(action)
        env.render()
        
        if done:
            if reward == 1:
                print("Agent won!")
            elif reward == 0.5:
                print("Draw!")
            break
