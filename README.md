# Q-Learning on FrozenLake

表格型 Q-Learning 在 FrozenLake 上的实现，包含 8×8 与 12×12 两种地图的完整训练、日志记录与结果可视化。

实验目的、算法推导与结果分析见 [实验报告.md](实验报告.md)。

## 结构

| 路径 | 说明 |
|---|---|
| `config.py` | 超参数配置（学习率、折扣因子、探索率衰减等） |
| `frozen_lake_environment.py` | FrozenLake 环境实现（8×8 / 12×12 预设），自行实现，不依赖 gym |
| `q_learning.py` | Q-Learning 智能体与训练主循环 |
| `logger.py` | 训练日志记录 |
| `train_8x8.py` / `train_12x12.py` | 两种地图的训练入口 |
| `visualize_results.py` | 绘制奖励曲线、成功率曲线与策略网格 |
| `output_final_policy.py` | 输出训练得到的最终策略 |
| `draw_training_flowchart.py` | 绘制训练流程图 |
| `saved_models/` | 训练好的 Q 表与评估指标（`.pkl`） |
| `visualization_output/` | 训练曲线与策略可视化结果 |
| `3. 比较两种地图上的学习效果（单独创建一个Python文件）.md` | 两种地图对比实验的说明 |

## 运行

```bash
python train_8x8.py        # 训练 8×8 地图
python train_12x12.py      # 训练 12×12 地图
python visualize_results.py  # 可视化训练结果
```

依赖：`numpy`、`matplotlib`。

## 结果

`saved_models/` 中保存了训练好的 Q 表与指标，`visualization_output/` 中保存了奖励曲线、成功率曲线与最终策略网格图，可直接查看而无需重新训练。
