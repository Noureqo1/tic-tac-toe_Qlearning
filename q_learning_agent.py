import numpy as np
from collections import defaultdict
import random

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
    def get_action(self, state, valid_actions):
        if not valid_actions:
            return None
            
 
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
            

        q_values = [self.q_table[state][action] for action in valid_actions]
        max_q = max(q_values)

        best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)
        
    def learn(self, state, action, reward, next_state, next_valid_actions):
        if not next_valid_actions:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state][next_action] 
                           for next_action in next_valid_actions)
            

        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q)
            
    def decay_epsilon(self, decay_rate=0.995):
        self.epsilon = max(0.01, self.epsilon * decay_rate)
