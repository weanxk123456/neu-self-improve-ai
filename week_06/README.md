# Minimal AgentFlow + LoRA Flow-GRPO

面向课程第一阶段的可运行参考实现。一个共享 Qwen 模型、四个逻辑模块、真实搜索、仅训练 Planner 的 LoRA、三种统一评测模式。优先可读性，没有 Ray/vLLM/分布式服务依赖。

**交付的是代码，不是已经完成的论文复现结果。** 没有提供训练好的 7B adapter，也没有伪造 Accuracy。正式提交前必须在你的 GPU 服务器完成训练、全量评测，并理解每个文件。

## 1. 范围与论文差异（务必阅读）

论文：https://arxiv.org/abs/2510.05592
官方参考：https://github.com/lupantech/AgentFlow

| 项目 | 本实现 | 影响 |
|---|---|---|
| 基座 | `Qwen/Qwen2.5-7B-Instruct`，三种模式相同 | 与作业模型匹配 |
| 四模块 | Planner、Python Executor、LLM Verifier、LLM Generator；reason 子任务调用冻结 LLM | 保留模块化流程，但不是逐行移植官方实现 |
| 提示词/记忆 | 自写短提示词、结构化记录、字符/token 截断 | 与原版提示词、记忆机制不同 |
| 工具 | Serper Google 搜索摘要、Wikipedia extract、网页文本、冻结 LLM reason | 不含原版完整工具集及 LLM 搜索摘要器、Python Coder |
| 默认搜索 | Wikipedia，无 key；正式比较建议三种 AgentFlow 设置统一为 Serper | Wiki-only 与原论文 Google 搜索不同 |
| 训练语料 | `prepare.py` 下载独立 NQ 子集 | 不包含论文完整混合语料及训练规模 |
| 更新 | 每步一道题、组内多条完整轨迹、最终奖励广播到各轮 Planner、token PPO clipping + sampled KL | 最小串行 Flow-GRPO 实现；不是官方训练器的逐项等价移植 |
| 轨迹权重 | 每条轨迹同权，每轮同权，轮内 token 平均 | 明确的长度归一化选择，应与老师指定版本核对 |
| 训练参数 | 见 `run.py --help`；小规模 LoRA 起点 | 默认 context/输出长度/batch/LR/轮数不是论文完整配置 |
| 奖励/评分 | 默认 normalized EM；可选兼容 API 的语义 judge | EM 不是论文 GPT-4o Accuracy；judge prompt 也是简化版 |
| 试验次数 | 每命令一个 seed | 多次独立运行后自行报告均值，不能把单次结果当论文均值 |
| 量化 | 默认不量化；可选 `--four-bit` | 开启即 QLoRA/量化推理，必须披露 |

因此这是**功能完整的最小再实现**，不是“运行后保证得到 12.0/58.4/69.6”的承诺。若老师要求严格数值复现，需要进一步对齐官方工具、提示词、数据与预算。不要把只跑几步的 smoke run 当成训练成功。

## 2. 安装（Linux GPU 服务器）

建议 Python 3.11、独立环境。先按服务器驱动安装兼容 CUDA PyTorch，再装依赖；仓库给出的依赖组合使用 PyTorch 2.6.0。

```bash
conda create -n agentflow-lora python=3.11 -y
conda activate agentflow-lora
cd agentflow_lora
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
python tests/integration_tiny.py
```

`integration_tiny.py` 在本地构造随机微型 Qwen，验证真实 LoRA 更新/冻结/保存，不下载权重，不代表语言能力。

如果使用量化：

```bash
pip install bitsandbytes==0.45.5
```

使用 7B 时建议先申请服务器 GPU。实际显存取决于 context、长度和量化，代码不承诺 12GB 能完成 7B 在线训练。CPU 可以执行微型测试，7B CPU 运行很慢。

### 你之前遇到的 home quota

把工程、模型缓存、pip 缓存和实验输出放在你有配额的 scratch；以下路径按 NEU 常见路径写，若集群不同请替换。

```bash
export TASK_ROOT="/scratch/$USER/agentflow_project"
mkdir -p "$TASK_ROOT/cache/huggingface" "$TASK_ROOT/cache/torch" "$TASK_ROOT/cache/pip"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export TORCH_HOME="$TASK_ROOT/cache/torch"
export PIP_CACHE_DIR="$TASK_ROOT/cache/pip"
```

在有 CUDA 资源的计算节点运行，不能只在无 GPU 的登录节点训练。程序为单卡实现；不要使用 `torchrun` 启动多个重复进程。

## 3. 准备数据

```bash
python prepare.py --train-limit 2000
```

数据来源为 FlashRAG 的 Bamboogle/NQ 标准化 JSONL（保留原问题与答案）。

得到 `data/bamboogle.jsonl`（完整下载的测试集）和 `data/train.jsonl`（独立 NQ 子集）。需要 Hugging Face 网络访问；脚本对未知 schema 显式报错，不会悄悄使用 toy 数据。已有目标文件则拒绝覆盖。

如果服务器无法访问 Hugging Face，可以先在可联网电脑下载以下两份文件，再上传到服务器；不要用测试集充当训练集：

- https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl
- https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/nq/train.jsonl

```bash
python prepare.py --test-source /path/to/test.jsonl --train-source /path/to/train.jsonl --train-limit 2000
```

本地自有数据也可直接用 JSONL：

```json
{"id":"1","question":"Your question?","answers":["answer","alias"]}
```

`data/example_train.jsonl` / `example_test.jsonl` 只供流程检查，绝不是 Bamboogle。不能用 Bamboogle 答案训练或选择超参数。训练必须提供 `--heldout`，程序自动检查规范化问题重叠；这不能替代完整的来源/泄漏审查。

## 4. 搜索与 judge

### 默认免费配置

`--search wiki --judge em` 可不设置 API key。模型/数据下载及 Wikipedia 仍需要网络。

### 更接近论文的搜索和语义评测

```bash
export SERPER_API_KEY='填写自己的搜索API key'
export JUDGE_API_KEY='填写自己的judge API key'
export JUDGE_BASE_URL='https://api.openai.com/v1'
```

`JUDGE_BASE_URL` 是兼容 Chat Completions 的 API 根路径，包含 `/v1`，不包含 `/chat/completions`。支持其他兼容服务，但应在报告披露模型名称。judge 输出布尔正确性，按语义判断，不负责回答问题。使用外部服务会产生相应费用。key 从环境变量读取，不会写入配置/源码。

`search` 在 Serper 模式使用 Google 结果；`wiki` 始终是 Wikipedia；`read` 获取普通网页文本。工具报错会成为观察，允许 Planner 尝试恢复。成功结果按查询缓存，错误不缓存。若搜索持续失败，请先检查 key、配额、出口网络；代码不能弥补服务器外网阻断。

## 5. 运行顺序

所有命令在解压后的 `agentflow_lora` 目录执行。以下以 Serper + GPT-4o judge 为例；无需 key 的调试可替换为 `--search wiki --judge em`。**不要跨设置混用评分规则后直接比较 Accuracy。**

如果只想先执行完整流程，且已准备数据，可运行：

```bash
bash run_all.sh
# 或在配置 API key 后：
SEARCH=serper JUDGE=llm STEPS=100 bash run_all.sh
```

脚本依次跑三行评测和训练，默认不量化。需要自定义显存/采样/LoRA 参数时使用下面的独立命令。

### A. 未训练直接回答

```bash
python run.py evaluate --mode direct \
  --data data/bamboogle.jsonl --judge llm --judge-model gpt-4o \
  --out runs/direct_seed42 --seed 42
```

### B. 未训练 AgentFlow

```bash
python run.py evaluate --mode agentflow \
  --data data/bamboogle.jsonl --search serper \
  --judge llm --judge-model gpt-4o --turns 3 \
  --out runs/agentflow_seed42 --seed 42
```

先检查单题：

```bash
python run.py demo --mode agentflow --search serper \
  --question 'Who was president of the United States when Citibank was founded?'
```

`demo` 展示动作、工具观察、Verifier、最终答案，不使用 gold。

### C. LoRA Flow-GRPO 训练

先用 2 步检查代码和资源；输出目录必须是新的：

```bash
python run.py train --data data/train.jsonl --heldout data/bamboogle.jsonl \
  --search serper --judge llm --judge-model gpt-4o \
  --steps 2 --group 4 --turns 3 --save-every 1 \
  --out runs/train_smoke
```

再启动独立的正式实验（100 步只是起点，不保证足够）：

```bash
python run.py train --data data/train.jsonl --heldout data/bamboogle.jsonl \
  --search serper --judge llm --judge-model gpt-4o \
  --steps 100 --group 4 --turns 3 --temperature 0.5 \
  --context 2048 --max-new 256 --rank 16 --alpha 32 \
  --lr 1e-5 --beta 0.001 --save-every 20 \
  --out runs/train_main --seed 42
```

显存不足可以在**相应训练及各评测命令**加 `--four-bit`，并统一记录量化设置；先减小 context/输出长度，注意降低预算也改变实验条件。

一组 reward 全相同时 advantage 为 0，本实现跳过该组更新。检查 `train.jsonl` 的 `updated`、`optimizer_updates`、`grad_norm`，不能只看程序结束。`status.json` 会提醒零更新。长期全错/全对需要检查数据难度、工具质量、group size，而不是伪造奖励或在测试集调参。

每步存储完整轨迹和 token IDs，便于解释重放。中间 checkpoint 是 LoRA adapter，不包含 7B 基座。`--adapter` 可用于从已有 adapter 继续新的训练，但**不恢复 optimizer/RNG**，不属于严格断点续训。

### D. 训练后 AgentFlow

```bash
python run.py evaluate --mode flow-grpo \
  --adapter runs/train_main/adapter_final \
  --data data/bamboogle.jsonl --search serper \
  --judge llm --judge-model gpt-4o --turns 3 \
  --out runs/flow_grpo_seed42 --seed 42

python run.py demo --mode flow-grpo --search serper \
  --adapter runs/train_main/adapter_final \
  --question 'Who was president of the United States when Citibank was founded?'

python summarize.py runs/direct_seed42 runs/agentflow_seed42 runs/flow_grpo_seed42
```

评测第二/三行保持模型、工具、context、轮数、量化与 judge 相同，区别主要是 Planner adapter。默认评测贪心生成；改变 seed 未必产生变化，外部服务也不保证完全确定。需要复现论文的随机试验协议时另行对齐并记录采样配置。

## 6. 输出和故障恢复

评测生成 `config.json`、逐题 `predictions.jsonl`、最终 `metrics.json`。只有全部完成才生成 metrics；若中途失败，已有逐题结果保留，但不自动续跑。修复后使用新的输出目录重跑，工具/judge 缓存可复用。

训练生成 `train.jsonl`、`trajectories/`、`adapter_step_N/`、`adapter_final/`、`status.json`。若中断，可从最近 adapter 启动新的训练，但 optimizer 会重置。不要把失败/部分评测当作全量成绩。

## 7. 逐行讲解重点

- `core.py`：四模块控制流；memory 只包含动作、观察和验证；gold 不进入 Agent。
- `model.py`：共享基座；`disable_adapter()` 隔离冻结模块；保留真实 prompt/action token IDs；logits 位置 `len(prompt)-1` 开始预测第一个 action token。
- `tools.py`：真实 HTTP、搜索后端和缓存；不会根据答案查表。
- `judge.py`：奖励计算，与解题模型分离；EM 和语义 judge 区别。
- `train.py`：组内完整 rollout，优势标准化，old/ref logprob，ratio/clipping/KL，仅 Planner token 反向传播。
- `run.py`：三种模式和统一评测；`prepare.py`：独立训练/测试来源。

训练目标（最小实现约定）：

```text
A_i = (reward_i - group_mean) / (group_population_std + 1e-8)
r_t = exp(log_pi_current(action_token | stored_context)
          - log_pi_old(action_token | stored_context))
KL_t = exp(log_pi_ref - log_pi_current) - (log_pi_ref - log_pi_current) - 1
loss_turn = mean_tokens[-min(r_t*A_i, clip(r_t,1-eps,1+eps)*A_i) + beta*KL_t]
loss_group = mean_trajectories(mean_turns(loss_turn))
```

采样使用 temperature=0.5、无 top-k/top-p 截断；训练 logprob 同样除以 temperature，避免 rollout/更新分布不一致。参考策略为禁用 LoRA 的基座。每组只做一次 optimizer step，所以初次前向 ratio 接近 1，裁剪通常尚不激活，但策略梯度仍存在；不要误以为 ratio=1 就没有学习。梯度更新前缓存 old/ref，之后才反向和 optimizer.step。

5–10 分钟演示：三行结果表 → 单题完整轨迹 → `Agent.run` → `log_probs` token mask → `train.py` loss → adapter checkpoint/实际更新次数 → 如实说明差异。

## 8. 提交仓库

不需要提交 7B 模型缓存、API key 或整个虚拟环境。保留源码、README、依赖、实际实验配置、三行 metrics，以及少量能说明问题的真实轨迹。

```bash
git init
git add .
git commit -m 'Minimal AgentFlow with planner LoRA Flow-GRPO'
```

`.gitignore` 默认忽略 runs/data/cache。如果要提交你选择的结果，可将脱敏后的 metrics 和配置复制到 `results/` 再提交。请保留原始训练记录以供老师核验。
