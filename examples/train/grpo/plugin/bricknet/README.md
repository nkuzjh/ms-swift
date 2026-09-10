# BrickNet GRPO plugin

本目录是 BrickNet reward、推理和评测脚本的代码旁入口，不是独立项目文档。

阅读顺序：

1. [项目文档入口](../../../../../../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/README.md)
2. [Constructor Plan](../../../../../../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Constructor%20Plan.md)
3. [ms-swift record.md](../../../../../record.md) — exp0 配置与可直接执行命令
4. [统一结果账本](../../../../../experiment_results.md)

当前正式入口：

- `prepare_exp3_base.sh`：准备 PT-exp0 merged base；
- `grpo_exp0_qwen35_08b_exp3.sh`：单轮 GRPO；
- `infer_exp0_qwen35_08b_exp3.sh`：交互推理；
- `evaluate_exp0_qwen35_08b_exp3.sh`：统一 VAL512 评测；
- `reward_plugin.py` / `evaluate_experiment.py`：reward 与评测实现。

exp0 使用 parse、inventory、length、collision-prefix 和 pose 五项 reward；权重与采样配置以 `record.md` 为准。`--dataset` 可显式选择 v1 或已物化的 official v2 JSONL；未传时保持脚本默认值。训练/评测 v2 必须使用独立输出身份，不能复用 v1 cache 或结果。

Stage 8/9 的协议、数据 gate 和 multi-turn 路线不在本 README 维护，见 Planner 的 [Stage 8 文档](../../../../../../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Stage%208%20ReAct%20Three-Source%20Auto-Annotation.md)。
