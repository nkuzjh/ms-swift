# BrickNet-MM GRPO exp0

该实验从 LlamaFactory 的 Qwen3.5-0.8B `exp3` 继续做 ms-swift GRPO。`exp3`
训练和评测时的有效权重是：

```text
Qwen/Qwen3.5-0.8B
+ PT-exp0 LoRA
+ exp3 SFT LoRA
```

ms-swift 的 GRPO policy/reference 各只接收一个 adapter，因此本配置先把 PT-exp0
合并进 base，再把 exp3 同时作为可训练 policy adapter 和冻结 reference adapter。
这不是把 exp3 当成独立于 PT-exp0 的 LoRA 使用。

## 1. 准备环境、数据和模型

在安装当前 ms-swift checkout 的 conda 环境中运行。当前机器环境名是 `swift`。

```bash
cd /home/jiahao/task/ms-swift
conda activate swift

python -m pip install -r examples/train/grpo/plugin/bricknet/requirements.txt

mkdir -p data
ln -s \
  /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-RL/samples/BrickNet-MM-RL_n2000_seed42.jsonl \
  data/BrickNet-MM-RL_n2000_seed42.jsonl

bash examples/train/grpo/plugin/bricknet/prepare_exp3_base.sh
```

仓库中已经创建了上述数据软链接。`prepare_exp3_base.sh` 可重复运行：已有完整的 PT merged
模型时会跳过导出。脚本还会为 PT-exp0 和 exp3 建立只包含根目录
`adapter_config.json`/`adapter_model.safetensors` 的干净 adapter view，防止 ms-swift 把
LlamaFactory 输出中的 `checkpoint-*` 子目录误加载成额外 adapter。

正式碰撞 reward 需要：

- `meshlib`
- `BRICKNET_DATA=/home/jiahao/.local/share/bricknet`
- `${BRICKNET_DATA}/inset/*.ply`

本机该 mesh 目录包含 21,084 个文件。不要在正式实验中设置
`BRICKNET_COLLISION_CHECK=false`。

## 2. 命令行训练

推荐直接运行可版本化 YAML：

```bash
cd /home/jiahao/task/ms-swift
conda activate swift
bash examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.sh
```

脚本等价于：

```bash
swift rlhf examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.yaml
```

额外的 ms-swift 参数可以追加在脚本后，例如只做参数/数据小规模诊断：

```bash
bash examples/train/grpo/plugin/bricknet/grpo_exp0_qwen35_08b_exp3.sh \
  --max_steps 1 \
  --save_steps 1
```

五项 reward 与 hard mining 完全对齐：

| Reward function | 权重 | 定义 |
| --- | ---: | --- |
| `bricknet_parse_prefix` | 0.20 | 完整可解析为 1；否则为已解析 parts / target parts |
| `bricknet_inventory_f1` | 0.20 | `(part_id, color_code)` multiset F1 |
| `bricknet_length` | 0.10 | `min(pred_parts,target_parts)/max(...)` |
| `bricknet_collision_prefix` | 0.20 | 无碰撞为 1；否则为首次碰撞 index / target parts |
| `bricknet_pose_match` | 0.30 | 全局刚体对齐后匹配的 target pose 比例 |

插件还注册了两个未默认启用的函数：

- `bricknet_dense_reward`：上述五项的单函数加权和。
- `bricknet_strict_success`：完整解析、长度和库存精确、无碰撞且 pose match 达阈值时为 1。

不要同时使用五项加权 reward 和 `bricknet_dense_reward`，否则会重复计入同一奖励。

## 3. 推理

训练结束并释放训练所占 GPU 后运行：

```bash
cd /home/jiahao/task/ms-swift
conda activate swift
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

脚本默认从
`output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8/` 自动选择数值最大的
`checkpoint-*`。正式记录结果时应固定检查点，例如：

```bash
EXP0_CHECKPOINT=/home/jiahao/task/ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8/checkpoint-1000 \
bash examples/train/grpo/plugin/bricknet/infer_exp0_qwen35_08b_exp3.sh
```

脚本执行的核心命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=589824 \
MIN_PIXELS=1024 \
swift infer \
  --model /home/jiahao/task/ms-swift/models/Qwen3.5-0.8B-PT-exp0-merged \
  --adapters /home/jiahao/task/ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8/checkpoint-1000 \
  --infer_backend transformers \
  --stream true \
  --temperature 0 \
  --max_new_tokens 4096 \
  --enable_thinking false \
  --response_prefix "" \
  --add_non_thinking_prefix false
```

GRPO checkpoint 保存的是从 SFT-exp3 adapter 初始化并继续更新后的完整 LoRA adapter，
因此推理时只加载该 checkpoint，不能再叠加
`models/Qwen3.5-0.8B-exp3-adapter`。进入交互界面后，在 query 中加入 `<image>`，再按提示
输入对应图片路径。

## 4. 完整评测

exp0 的固定评测入口会在 512 条 `BrickNet-MM_VAL` 上先批量推理，再依次计算：

- 与 LlamaFactory 相同实现的 BLEU-4、ROUGE-1/2/L；
- BrickNet 完整解析率、无碰撞率、有效前缀等结构指标；
- LDR 八视图渲染后的 PE、SigLIP2、VQAScore 及 coverage-adjusted 指标；
- 与 GRPO/hard-mining 同定义的 ParsePrefix、Inventory F1、Length、
  CollisionPrefix、PoseMatch、稠密 reward、strict success 和分组指标。

```bash
cd /home/jiahao/task/ms-swift
conda activate swift
bash examples/train/grpo/plugin/bricknet/evaluate_exp0_qwen35_08b_exp3.sh
```

正式设置为 `max_new_tokens=4096, temperature=1.0, top_p=0.95, top_k=20,
seed=42`，与 LlamaFactory exp3 的验证集生成口径一致。结果写到：

```text
output/bricknet_eval/grpo_exp0_qwen35_08b_exp3_val_t1_p095_k20/
├── swift_inference.jsonl  # ms-swift 原始批量推理结果
├── scored.jsonl           # BrickNet 逐样本解析/碰撞结果
├── alignment.jsonl        # 逐样本 GRPO 对齐指标
├── metrics.json           # 所有汇总指标（机器可读）
├── metrics.md             # 所有汇总指标（表格）
└── run_manifest.json      # checkpoint、输入哈希、采样设置和运行状态
```

后续实验直接调用通用入口，更换 checkpoint、实验名和输出目录即可：

```bash
python examples/train/grpo/plugin/bricknet/evaluate_experiment.py run \
  --experiment-name <name> \
  --checkpoint <adapter-or-full-checkpoint> \
  --base-model <adapter对应的base-model> \
  --output-dir output/bricknet_eval/<name>
```

若 checkpoint 是完整模型目录可省略 `--base-model`；若已经有 ms-swift
`swift infer --result_path` 生成的结果，可用 `--inference-results <jsonl>` 跳过推理。
默认会复用已完成且输入身份一致的阶段；换采样参数或 checkpoint 时应使用新的输出目录。

## 5. Web-UI 设置

先启动：

```bash
cd /home/jiahao/task/ms-swift
conda activate swift

NO_PROXY=localhost,127.0.0.1,0.0.0.0 \
no_proxy=localhost,127.0.0.1,0.0.0.0 \
swift web-ui --lang zh
```

打开「LLM GRPO」，设置以下可见字段：

| 页面区域 / 字段 | 值 |
| --- | --- |
| 模型 id 或路径 | `/home/jiahao/task/ms-swift/models/Qwen3.5-0.8B-PT-exp0-merged` |
| 模型类型 | `qwen3_5` |
| 数据集名称 | `/home/jiahao/task/ms-swift/data/BrickNet-MM-RL_n2000_seed42.jsonl` |
| 验证集拆分比例 | `0` |
| 句子最大长度 | `4096` |
| 奖励函数 | 依次填写五项 `bricknet_parse_prefix bricknet_inventory_f1 bricknet_length bricknet_collision_prefix bricknet_pose_match` |
| 奖励函数权重 | `0.20 0.20 0.10 0.20 0.30` |
| 外部插件文件 | `/home/jiahao/task/ms-swift/examples/train/grpo/plugin/bricknet/bricknet_reward_plugin.py` |
| 训练方式 | `lora` |
| LoRA rank / alpha / dropout | `64 / 128 / 0` |
| 冻结 ViT / aligner | 均勾选 |
| 训练精度 | `bfloat16` |
| 训练 batch size | `2` |
| 梯度累计步数 | `8` |
| 学习率 | `5e-6` |
| Epoch | `1` |
| 保存步数 | `50` |
| 输出目录 | `/home/jiahao/task/ms-swift/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8` |
| KL 系数 beta | `0.04` |
| 记录生成内容 | 勾选 |
| 使用 vLLM / 模式 | 勾选 / `colocate` |
| GPU 显存利用率 | `0.45` |
| TP / 最大模型长度 / sleep | `1 / 9216 / 1` |
| 采样数量 G | `8` |
| 最大生成长度 | `4096` |
| temperature / top-p | `0.9 / 1.0` |

Web-UI 的 top-k 滑块不能填写 `-1`，而且页面没有 policy/reference adapter 输入框。把下面
JSON 原样填入「更多参数 → 其他参数设置」：

```json
{
  "adapters": ["/home/jiahao/task/ms-swift/models/Qwen3.5-0.8B-exp3-adapter"],
  "ref_adapters": ["/home/jiahao/task/ms-swift/models/Qwen3.5-0.8B-exp3-adapter"],
  "top_k": -1,
  "repetition_penalty": 1.0,
  "response_prefix": "",
  "add_non_thinking_prefix": false,
  "enable_thinking": false,
  "max_pixels": 589824,
  "truncation_strategy": "delete",
  "target_modules": ["all-linear"],
  "vllm_max_lora_rank": 64,
  "vllm_enable_lora": true,
  "scale_rewards": "group",
  "overlong_filter": true,
  "lr_scheduler_type": "cosine",
  "warmup_steps": 0,
  "max_grad_norm": 1.0,
  "save_total_limit": 3,
  "logging_steps": 1,
  "report_to": ["tensorboard"],
  "dataset_shuffle": true,
  "data_seed": 42
}
```

「环境变量」填写一行（各项用空格分隔，值中不能包含空格）：

```text
ROOT_IMAGE_DIR=/home/jiahao/task/BrickNet BRICKNET_ROOT=/home/jiahao/task/BrickNet BRICKNET_DATA=/home/jiahao/.local/share/bricknet BRICKNET_COLLISION_CHECK=true BRICKNET_POSE_TRANSLATION_TOLERANCE=0.5 BRICKNET_POSE_ROTATION_TOLERANCE=5.0 BRICKNET_POSE_SUCCESS_THRESHOLD=1.0 MAX_PIXELS=589824 MIN_PIXELS=1024 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

先勾选「仅生成运行命令」，检查生成命令中确实同时存在 `--adapters`、
`--ref_adapters`、五个 reward 名称和对应权重，再取消 dry-run 开始训练。

## 6. Reward smoke test

```bash
BRICKNET_ROOT=/home/jiahao/task/BrickNet \
BRICKNET_DATA=/home/jiahao/.local/share/bricknet \
python examples/train/grpo/plugin/bricknet/verify_reward.py
```

脚本取数据集第一行，用 reference solution 作为 completion。五项 reward、strict success
和总 reward 都必须为 `1.0`。
