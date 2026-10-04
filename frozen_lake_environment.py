import gymnasium as gym

class EnvironmentInteractionWrapper:
    """
    环境交互接口包装器
    将 gymnasium 环境的 reset 和 step 方法包装为智能体所需的格式
    """
    def __init__(self, env):
        self.env = env

    def reset(self):
        state, info = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

def create_preset_8x8_frozen_lake_environment():
    """
    创建预设的 8x8 FrozenLake 环境，并启用随机滑动条件
    """
    environment = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=True
    )
    return EnvironmentInteractionWrapper(environment)

def create_custom_12x12_frozen_lake_environment():
    """
    创建自定义的 12x12 FrozenLake 环境，通过 desc 传入自定义地图，并启用随机滑动条件
    """
    custom_12x12_map_desc = [
        "SFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFHFFFFHFFF",
        "FFFFFHFFFFFF",
        "FFFHFFFFFFFF",
        "FFFFFFFHFFFF",
        "FHFFFFFFFFFF",
        "FFFFHFFFFFFF",
        "FFFFFFFFHFFF",
        "FFFHFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFG"
    ]
    
    environment = gym.make(
        "FrozenLake-v1",
        desc=custom_12x12_map_desc,
        is_slippery=True
    )
    return EnvironmentInteractionWrapper(environment)
