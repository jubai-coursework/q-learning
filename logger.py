import os
import pickle
import numpy as np

from config import Config

class TrainingLogger:
    def __init__(self, save_dir=Config.SAVE_DIR):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_results(self, map_name, agent, rewards, steps):
        """
        保存训练后的 Q 表和相关指标
        """
        # 保存 Q 表
        q_table_path = os.path.join(self.save_dir, f"{map_name}_q_table.pkl")
        with open(q_table_path, "wb") as f:
            pickle.dump(agent.q_table, f)
        
        # 保存奖励和步数历史
        metrics_path = os.path.join(self.save_dir, f"{map_name}_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump({"rewards": rewards, "steps": steps}, f)
        
        print(f"日志：[{map_name}] 训练结果已保存到 {self.save_dir} 目录下。")

    def load_results(self, map_name):
        """
        加载保存的 Q 表和指标
        """
        q_table_path = os.path.join(self.save_dir, f"{map_name}_q_table.pkl")
        metrics_path = os.path.join(self.save_dir, f"{map_name}_metrics.pkl")

        if not os.path.exists(q_table_path) or not os.path.exists(metrics_path):
            raise FileNotFoundError(f"找不到 {map_name} 的模型或训练数据，请先运行训练脚本！")

        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)
        
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

        return q_table, metrics["rewards"], metrics["steps"]
