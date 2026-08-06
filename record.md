# notes

- 项目：ms-swift 上的 BrickNet-MM SFT / GRPO 实验。
- 唯一总推进计划：`/home/jiahao/task/BrickNet/BrickNet-MM Agentic LEGO Planner/Constructor Plan.md`。
- 2026-08-05 决策：当前 exp0 是基础 2k、单轮 completion GRPO；不是 ReAct multi-turn GRPO。
  policy-specific full hard mining 已暂停，默认从 66,456 条基础池逐级扩容；条件性恢复 mining 时最多 2,000 prompts。
- 统一实验进度与结果：[experiment_results.md](experiment_results.md)。
- 上游实验记录：`/home/jiahao/task/LlamaFactory/record.md`。
- 上游 BrickNet RL 说明：`/home/jiahao/task/BrickNet/data_preprocess/GRPO_DATASET_zh.md`。
- 当前主模型：`Qwen/Qwen3.5-0.8B`。
- LlamaFactory `exp3` 的有效初始策略是两层 LoRA 的组合：
  1. `train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64`
  2. `train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64`
- ms-swift GRPO 只支持一个 policy/reference adapter。为复现上述组合，先将 PT-exp0
  合并到 base model，再把 exp3 同时传给 `--adapters` 与 `--ref_adapters`。
- Qwen3.5 使用非 thinking 输出：`--enable_thinking false --response_prefix ''
  --add_non_thinking_prefix false`，对应 LlamaFactory 的 `qwen3_5_nothink`。
- BrickNet 图片路径相对 BrickNet 仓库根目录，训练脚本显式设置
  `ROOT_IMAGE_DIR=/home/jiahao/task/BrickNet`。
- 本次链路验证使用 `/home/jiahao/miniconda3/envs/swift`：ms-swift `4.5.0.dev0`、
  PyTorch `2.11.0+cu130`、Transformers `5.12.1`、PEFT `0.19.1`、TRL `0.29.1`、
  vLLM `0.26.0`。

# experiments

## exp0

### Qwen3.5-0.8B-PT-SFT-exp3 GRPO

- 初始化：Qwen3.5-0.8B + 已合并 PT-exp0 + exp3 SFT LoRA。
- 数据：`BrickNet-MM-RL_n2000_seed42.jsonl`，2,000 个 prompt，每个 prompt 在线采样 8 个
  completion。
- 算法：GRPO，LoRA 后训练。
- rollout：`temperature=0.9`、`top_p=1.0`、关闭 top-k、最长 completion 4096 tokens。
- reward：

  | 名称 | 权重 | 简要含义 |
  | --- | ---: | --- |
  | `bricknet_parse_prefix` | 0.20 | 完整可解析得 1；解析失败时按已解析 part 占目标 part 的比例给分 |
  | `bricknet_inventory_f1` | 0.20 | 生成结果与目标的 `(part_id, color_code)` 多重集合 F1 |
  | `bricknet_length` | 0.10 | 生成 part 数与目标 part 数的接近程度，数量一致得 1 |
  | `bricknet_collision_prefix` | 0.20 | 无碰撞得 1；有碰撞时按首次碰撞前完成的比例给分 |
  | `bricknet_pose_match` | 0.30 | 全局刚体对齐后，位置和旋转满足容差的目标 part 比例 |

  总 reward 为五项的加权和：
  `0.20*parse + 0.20*inventory + 0.10*length + 0.20*collision + 0.30*pose`，
  取值范围为 `[0, 1]`。当前使用 `scale_rewards=group`，对同一 prompt 的 8 个
  completion 做组内相对归一化。`KL` 由 `beta=0.04` 在 loss 中单独约束，不计入日志中的
  `reward`；插件注册的 `bricknet_dense_reward` 和 `bricknet_strict_success` 未在 exp0
  中启用。
- pose 容差：平移 0.5、旋转 5 度。
- 配置目录：`examples/train/grpo/plugin/bricknet/`。

**prepare**

```bash
bash examples/train/grpo/plugin/bricknet/prepare_exp3_base.sh
```

**train**

```bash
bash examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.sh
```

**infer**

训练结束并释放训练所占 GPU 后运行。默认自动加载 exp0 数值最大的 `checkpoint-*`：

```bash
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

需要固定检查点以复现实验时显式指定，例如最终的 `checkpoint-1000`：

```bash
EXP0_CHECKPOINT=/home/jiahao/task/ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8/checkpoint-1000 \
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

该命令以 PT-exp0 merged model 为 base，仅加载 GRPO checkpoint 中已经包含 SFT-exp3
初始权重及 GRPO 更新后的 LoRA，不要再额外加载原始 SFT-exp3 adapter。进入交互界面后，
在 query 中使用 `<image>`，再按提示输入图片路径。

**evaluate**

固定使用 512 条 `BrickNet-MM_VAL` 和 LlamaFactory exp3 的正式采样口径
（`max_new_tokens=4096, temperature=1.0, top_p=0.95, top_k=20, seed=42`）：

```bash
bash examples/train/grpo/plugin/bricknet/evaluate_exp0_qwen35_08b_exp3.sh
```

该入口自动完成批量推理、LlamaFactory 口径 BLEU/ROUGE、BrickNet
解析/碰撞/LDR 八视图/PE/SigLIP2/VQAScore，以及与 GRPO reward 同定义的五项对齐指标、
dense reward 和 strict success。汇总结果位于：

```text
output/bricknet_eval/grpo_exp0_qwen35_08b_exp3_val_t1_p095_k20/metrics.json
output/bricknet_eval/grpo_exp0_qwen35_08b_exp3_val_t1_p095_k20/metrics.md
```

**web-ui**

```bash
NO_PROXY=localhost,127.0.0.1,0.0.0.0 \
no_proxy=localhost,127.0.0.1,0.0.0.0 \
swift web-ui --lang zh
```

Web-UI 本身不提供 YAML 导入。在「LLM GRPO」页面按
`examples/train/grpo/plugin/bricknet/README.md` 的字段表填写；同目录 YAML 是命令行的
等价、可版本化配置。
