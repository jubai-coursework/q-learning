# config.py

class Config:
    # 8x8 预设地图训练参数
    MAP_8x8_PARAMS = {
        "learning_rate": 0.05,
        "discount_factor": 0.99,
        "initial_exploration_rate": 1.0,
        "min_exploration_rate": 0.01,
        "exploration_decay_rate": 0.9999,
        "num_episodes": 60000
    }

    # 12x12 自定义地图训练参数
    MAP_12x12_PARAMS = {
        "learning_rate": 0.05,
        "discount_factor": 0.995,
        "initial_exploration_rate": 1.0,
        "min_exploration_rate": 0.005,
        "exploration_decay_rate": 0.999,
        "num_episodes": 100000
    }

    # 日志与数据保存路径
    SAVE_DIR = "saved_models"
