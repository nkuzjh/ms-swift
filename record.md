# ms-swift experiments

> 阅读前置：[项目文档入口](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/README.md) → [Constructor Plan](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Constructor%20Plan.md)。本文件只记录实验版本、简短配置/变动和可直接执行命令；结果见 [experiment_results.md](experiment_results.md)。

以下命令默认从 `/home/jiahao/task/ms-swift` 执行，并使用项目 `swift` 环境。

## exp0 — Qwen3.5-0.8B PT-SFT GRPO

配置：Qwen3.5-0.8B + 已合并 PT-exp0 base + exp3 SFT LoRA；2,000 prompts，每个 prompt 采样 8 completions；单轮 completion GRPO，不是 Stage 9 multi-turn。采样为 temperature 0.9、top-p 1、关闭 top-k、最长 4096 tokens；reward 权重为 parse/inventory/length/collision/pose = 0.20/0.20/0.10/0.20/0.30，KL beta 0.04。

### Prepare

```bash
cd /home/jiahao/task/ms-swift
bash examples/train/grpo/plugin/bricknet/prepare_exp3_base.sh
```

### Train

```bash
cd /home/jiahao/task/ms-swift
bash examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.sh
```

显式选择 official v2 RL-2k，只改变 dataset：

```bash
bash examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.sh \
  --dataset /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM_v2/sharegpt/BrickNet-MM-RL_n2000_seed42_v2.jsonl
```

### Infer

默认使用数值最大的 `checkpoint-*`：

```bash
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

固定 checkpoint：

```bash
EXP0_CHECKPOINT=/home/jiahao/task/ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8/checkpoint-1000 \
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

### Evaluate

```bash
bash examples/train/grpo/plugin/bricknet/evaluate_exp0_qwen35_08b_exp3.sh
```

使用 official v2 VAL 时必须使用独立实验名和输出目录：

```bash
bash examples/train/grpo/plugin/bricknet/evaluate_exp0_qwen35_08b_exp3.sh \
  --experiment-name grpo_exp0_qwen35_08b_exp3_val_v2_official \
  --dataset /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM_v2/sharegpt/BrickNet-MM_VAL_v2.jsonl \
  --output-dir output/bricknet_eval/grpo_exp0_qwen35_08b_exp3_val_v2_official
```

### WebUI

```bash
NO_PROXY=localhost,127.0.0.1,0.0.0.0 \
no_proxy=localhost,127.0.0.1,0.0.0.0 \
swift web-ui --lang zh
```

## 后续版本

Stage 8 R1-S/R1-C/R1-B 与 Stage 9 multi-turn GRPO 尚未成为本仓库可执行正式实验；对应 gate 见 [Stage 8 文档](../BrickNet/BrickNet-MM%20Agentic%20LEGO%20Planner/Stage%208%20ReAct%20Three-Source%20Auto-Annotation.md) 和总计划。命令真正落地后再在本文件新增版本，不预写运行结果。
