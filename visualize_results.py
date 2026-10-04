import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from logger import TrainingLogger

OUTPUT_DIR = "visualization_output"

def plot_training_reward_curve(rewards_history, window_size=None, title="Training Reward Curve", save_path=None):
    """
    可视化一：训练奖励曲线
    横轴：回合数，纵轴：每回合总奖励（结合滑动平均进行平滑处理）
    """
    if window_size is None:
        window_size = max(1, len(rewards_history) // 20)

    smoothed_rewards = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
    x_axis = range(window_size - 1, len(rewards_history))
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, smoothed_rewards, color='blue', alpha=0.8)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(f'Average Reward (Window: {window_size})')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()

def plot_success_rate_curve(rewards_history, chunk_size=100, title="Success Rate Curve", save_path=None):
    """
    可视化二：成功率曲线
    横轴：回合数，纵轴：每 chunk_size (默认100) 回合的成功比例
    """
    num_chunks = len(rewards_history) // chunk_size
    success_rates = []
    episodes = []

    for i in range(num_chunks):
        chunk = rewards_history[i*chunk_size : (i+1)*chunk_size]
        success_rate = sum(1 for r in chunk if r > 0) / chunk_size
        success_rates.append(success_rate)
        episodes.append((i + 1) * chunk_size)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, success_rates, marker='o', color='orange', linestyle='-')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()

def plot_final_policy_grid(q_table, grid_size, title="Final Policy Grid", save_path=None):
    """
    可视化三：最终策略网格图
    将策略以箭头形式绘制在图形化网格中
    """
    action_to_arrow = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    for state in range(len(q_table)):
        row = state // grid_size
        col = state % grid_size

        # Q表初始化为全1，若该状态所有Q值仍为1则未被探索过
        if np.allclose(q_table[state], 1.0):
            arrow = '*'
            color = 'gray'
        else:
            best_action = np.argmax(q_table[state])
            arrow = action_to_arrow[best_action]
            color = 'darkblue'

        y_pos = row + 0.5
        x_pos = col + 0.5
        ax.text(x_pos, y_pos, arrow, ha='center', va='center', fontsize=20, color=color)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.grid(color='black', linestyle='-', linewidth=1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)

    plt.title(title)
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = TrainingLogger()

    try:
        print("正在从日志直接加载模型数据并进行可视化展示...")

        # 8x8 地图
        q_table_8x8, rewards_8x8, steps_8x8 = logger.load_results("8x8")

        print("绘制 8x8 预设地图结果...")
        plot_training_reward_curve(rewards_8x8,
                                   title="8x8 Map: Training Reward Curve",
                                   save_path=os.path.join(OUTPUT_DIR, "8x8_reward_curve.png"))
        plot_success_rate_curve(rewards_8x8, chunk_size=100,
                                title="8x8 Map: Success Rate Curve",
                                save_path=os.path.join(OUTPUT_DIR, "8x8_success_rate.png"))
        plot_final_policy_grid(q_table_8x8, 8,
                               title="8x8 Map: Final Policy Grid",
                               save_path=os.path.join(OUTPUT_DIR, "8x8_policy_grid.png"))

        # 12x12 地图
        q_table_12x12, rewards_12x12, steps_12x12 = logger.load_results("12x12")

        print("绘制 12x12 自定义地图结果...")
        plot_training_reward_curve(rewards_12x12,
                                   title="12x12 Map: Training Reward Curve",
                                   save_path=os.path.join(OUTPUT_DIR, "12x12_reward_curve.png"))
        plot_success_rate_curve(rewards_12x12, chunk_size=100,
                                title="12x12 Map: Success Rate Curve",
                                save_path=os.path.join(OUTPUT_DIR, "12x12_success_rate.png"))
        plot_final_policy_grid(q_table_12x12, 12,
                               title="12x12 Map: Final Policy Grid",
                               save_path=os.path.join(OUTPUT_DIR, "12x12_policy_grid.png"))

        print(f"\n全部可视化图表已保存至 {OUTPUT_DIR}/ 目录。")

    except FileNotFoundError as e:
        print(e)
        print("提示：请先运行 train_8x8.py 和 train_12x12.py 来生成并保存训练结果！")

