import numpy as np
import random

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.99, initial_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.995):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = initial_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        return np.ones((self.num_states, self.num_actions))

    def choose_action(self, state):
        r = random.uniform(0, 1)
        if r < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_value(self, state, action, reward, next_state, done):
        max_next_q_value = np.max(self.q_table[next_state, :]) if not done else 0.0
        td_target = reward + self.discount_factor * max_next_q_value
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

def train_agent(agent, env, num_episodes):
    rewards_history = []
    successful_steps_history = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps_taken = 0
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps_taken += 1

        agent.decay_exploration_rate()

        rewards_history.append(episode_reward)
        if episode_reward > 0:
            successful_steps_history.append(steps_taken)

    return agent, rewards_history, successful_steps_history

def test_agent(agent, env, num_episodes):
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = 0.0

    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    agent.exploration_rate = original_exploration_rate
    return total_rewards
