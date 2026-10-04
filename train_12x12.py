import numpy as np
from q_learning import QLearningAgent, train_agent
from frozen_lake_environment import create_custom_12x12_frozen_lake_environment
from config import Config
from logger import TrainingLogger

def analyze_training_metrics(rewards_history, successful_steps_history):
    average_return = np.mean(rewards_history)
    success_rate = np.mean([1 if r > 0 else 0 for r in rewards_history])
    average_steps = np.mean(successful_steps_history) if len(successful_steps_history) > 0 else 0.0
    return average_return, success_rate, average_steps

def run_experiment(map_name, env_func, params, logger):
    print(f"开始在 {map_name} 地图上进行训练...")
    env = env_func()
    agent = QLearningAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        learning_rate=params['learning_rate'],
        discount_factor=params['discount_factor'],
        initial_exploration_rate=params['initial_exploration_rate'],
        min_exploration_rate=params['min_exploration_rate'],
        exploration_decay_rate=params['exploration_decay_rate']
    )
    
    agent, rewards, steps = train_agent(agent, env, params['num_episodes'])
    logger.save_results(map_name, agent, rewards, steps)
    
    avg_ret, succ_rate, avg_steps = analyze_training_metrics(rewards, steps)
    
    print(f"[{map_name} 训练完成]")
    print(f"  平均回报: {avg_ret:.4f}")
    print(f"  成功率:   {succ_rate:.2%}")
    print(f"  平均步数 (仅成功回合): {avg_steps:.2f} 步\n")
    
    return agent, rewards, steps

if __name__ == "__main__":
    logger = TrainingLogger()
    print("========== [开始 12x12 地图训练] ==========")
    run_experiment(
        "12x12", 
        create_custom_12x12_frozen_lake_environment, 
        Config.MAP_12x12_PARAMS, 
        logger
    )
