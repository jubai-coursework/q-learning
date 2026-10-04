import numpy as np
from logger import TrainingLogger

def generate_greedy_policy_string(q_table, grid_size):
    """
    根据给定的 Q表 和 地图尺寸 输出最终状态图的箭头策略。
    0: 左 ←
    1: 下 ↓
    2: 右 →
    3: 上 ↑
    """
    action_to_arrow = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }

    num_states = q_table.shape[0]
    policy_grid_lines = []

    for i in range(0, num_states, grid_size):
        row_symbols = []
        for j in range(grid_size):
            state = i + j
            
            # 检查是否所有Q值均为0（说明是未经探索的状态、洞或者终点）
            if np.all(q_table[state] == 0):
                row_symbols.append('*') # 用 '*' 表示全0状态，防止误导
            else:
                best_action = np.argmax(q_table[state])
                arrow = action_to_arrow.get(best_action, ' ')
                row_symbols.append(arrow)

        policy_grid_lines.append("  ".join(row_symbols))

    return "\n".join(policy_grid_lines)

def print_final_policy(q_table, grid_size, map_name="Default"):
    print(f"--- 【{map_name}】最终策略 ---")
    policy_grid_str = generate_greedy_policy_string(q_table, grid_size)
    print(policy_grid_str)
    print("--------------------------\n")

if __name__ == "__main__":
    logger = TrainingLogger()

    try:
        # 获取 8x8 地图的 Q 表并输出策略
        q_table_8x8, _, _ = logger.load_results("8x8")
        print_final_policy(q_table_8x8, 8, "预设 8x8 地图")

        # 获取 12x12 地图的 Q 表并输出策略
        q_table_12x12, _, _ = logger.load_results("12x12")
        print_final_policy(q_table_12x12, 12, "自定义 12x12 地图")

    except FileNotFoundError as e:
        print(e)
