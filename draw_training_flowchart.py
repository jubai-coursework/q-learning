import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体（Windows常用中文字体）
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')

    # 定义节点属性
    box_style = dict(boxstyle="round,pad=0.5", facecolor="#e0f7fa", edgecolor="#006064", linewidth=1.5)
    decision_style = dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", edgecolor="#f57f17", linewidth=1.5)
    
    nodes = {
        'start': ('开始训练任务\n(train_8x8.py / train_12x12.py)', (0.5, 0.95), box_style),
        'init_env': ('加载Config参数并初始化环境\nEnvironmentInteractionWrapper(env)', (0.5, 0.85), box_style),
        'init_agent': ('初始化 QLearningAgent \n(创建 Q-Table)', (0.5, 0.75), box_style),
        'ep_start': ('开始新 Episode\nenv.reset() 获取初始 state', (0.5, 0.65), box_style),
        'check_done': ('Episode 是否结束?\n(done == True)', (0.5, 0.53), decision_style),
        'choose_action': ('Agent选择动作\naction = choose_action(state)\n(ε-greedy 策略)', (0.5, 0.40), box_style),
        'env_step': ('环境执行动作\nnext_state, reward, done = env.step(action)', (0.5, 0.30), box_style),
        'update_q': ('更新 Q 值\nupdate_q_value(TD-Error计算)', (0.5, 0.20), box_style),
        'state_transit': ('状态转移\nstate = next_state', (0.5, 0.10), box_style),
        'ep_end': ('Episode结束，衰减探索率\ndecay_exploration_rate()\n记录日志 (logger.py)', (0.85, 0.53), box_style),
        'check_ep_max': ('达到最大 Episode 数?', (0.85, 0.75), decision_style),
        'end': ('保存 Q-Table 模型数据\n(saved_models/)', (0.85, 0.90), box_style),
    }

    # 绘制节点
    for key, (text, pos, style) in nodes.items():
        ax.text(pos[0], pos[1], text, ha='center', va='center', bbox=style, fontsize=10, zorder=3)

    # 绘制箭头引线
    def draw_arrow(pos_start, pos_end, label="", r_offset=0, up_offset=0):
        # 简单直接两点连线（如需折线可通过调整或分段绘制）
        ax.annotate(label,
                    xy=pos_end, xycoords='data',
                    xytext=pos_start, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5,
                                    connectionstyle=f"bar,fraction={r_offset}" if r_offset else "arc3"),
                    va='center', ha='center', fontsize=9, zorder=1)

    # 纵向主链路 (Episode 内)
    draw_arrow((0.5, 0.92), (0.5, 0.88))
    draw_arrow((0.5, 0.82), (0.5, 0.78))
    draw_arrow((0.5, 0.72), (0.5, 0.68))
    draw_arrow((0.5, 0.62), (0.5, 0.57))
    draw_arrow((0.5, 0.49), (0.5, 0.44), "否 (未结束)")
    draw_arrow((0.5, 0.36), (0.5, 0.33))
    draw_arrow((0.5, 0.27), (0.5, 0.23))
    draw_arrow((0.5, 0.17), (0.5, 0.13))
    
    # 状态转移回溯到是否结束判断 (折线)
    ax.annotate("", xy=(0.25, 0.53), xytext=(0.3, 0.10),
                arrowprops=dict(arrowstyle="-", color="#333333", lw=1.5, connectionstyle="angle,angleA=0,angleB=90"))
    ax.annotate("", xy=(0.5, 0.53), xytext=(0.25, 0.53),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5))
    ax.plot([0.3, 0.5], [0.10, 0.10], color="#333333", lw=1.5) # 连接 state_transit 左侧

    # Episode 结束分支
    draw_arrow((0.58, 0.53), (0.7, 0.53), "是 (Done)")
    
    # 从 Episode 结束到检查最大回合
    draw_arrow((0.85, 0.57), (0.85, 0.70))
    
    # 未达到最大回合，回到开始新 Episode
    draw_arrow((0.75, 0.75), (0.63, 0.65), "否")
    
    # 达到最大回合，结束保存
    draw_arrow((0.85, 0.80), (0.85, 0.87), "是")

    plt.title("Q-Learning Agent 训练过程逻辑流程图", fontsize=14, fontweight='bold', y=0.98)
    
    # 保存并展示
    plt.tight_layout()
    plt.savefig("draw_training_flowchart.png", dpi=100, bbox_inches='tight')
    print("流程图已保存至 draw_training_flowchart.png")

if __name__ == '__main__':
    draw_flowchart()
