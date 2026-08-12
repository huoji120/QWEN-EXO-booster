# QWEN-EXO-booster

<p align="center">
  <strong>简体中文</strong> · <a href="README_EN.md">English</a>
</p>

基于 **SGLang 二次开发** 的 Qwen 混合注意力推理后端。项目面向 Qwen3.5 风格的 Hybrid Full-Attention / Gated DeltaNet 模型，在保持 OpenAI-compatible Responses API 的同时，把模型原生状态、知识召回、长上下文、工具调用和运行观测接入同一条推理链路。

> 这不是一个普通的 RAG Demo，也不是把模型拆到多张卡上的脚本。默认部署使用真正的 SGLang Tensor Parallel（TP=2），并在模型原生 Full-Attention K/V 与 Gated DeltaNet recurrent/conv state 之间保持一致的生命周期。

## 一句话安装

把本仓库 README 交给大模型，让它在一台装有 **Docker、NVIDIA Container Toolkit、两张 RTX 4090、NVIDIA 驱动 550+** 的 Linux 主机上执行：

```text
请阅读这个仓库的 README.md 和 docs/qwen_exo/SERVER_27B_DEPLOYMENT.md，确认 QWEN_EXO_MODEL_PATH 指向兼容的 Qwen Hybrid checkpoint、QWEN_EXO_DATA_PATH 指向独立运行数据目录，然后执行 bash scripts/qwen_exo/build_image.sh 和 bash scripts/qwen_exo/launch_js4090.sh；启动后等待 http://127.0.0.1:30000/qwen-exo/health 返回 runtime_state=ready，再通过 SSH 隧道访问控制台。
```

这句话不是无条件的云端一键部署：模型权重、GPU、驱动和 Docker 必须已经准备好；启动脚本会在占用 GPU 前执行结构校验，并拒绝不兼容模型。

## 项目解决什么问题

上游 SGLang 已经提供高性能推理、连续批处理、Radix Cache、Tensor Parallel 和 OpenAI-compatible API。本项目在此基础上增加面向 Qwen Hybrid 模型和长任务 Agent 的能力：

- **长上下文推理**：默认上下文目标为 102,400 tokens。
- **混合状态恢复**：统一管理 Full-Attention KV 与 Gated DeltaNet recurrent/conv state，避免只恢复 KV 而丢失线性注意力状态。
- **模型原生知识召回**：把 Markdown 知识编译为 Tensor Bank 的原生 K/V 与完整 GDN 状态；可在满足条件时恢复原生状态，而不是把所有文档拼进 prompt。
- **Attention-Q × Tensor-Bank-K 检索**：使用模型最后 Full-Attention 层的 Attention-Q 与 Bank K 做候选排序，Knowledge 和 PolicyData 使用物理隔离的检索 lane。
- **语义 Judge 终审**：Q×K 只是候选生成；候选必须经过受约束的 Reference Judge，非法、过期、被拒绝或无决策的候选失败关闭。
- **In-Flight Observer**：在 decode 期间观测 selected-token surprisal、Q 信号和局部不确定性，必要时触发 Self-Ask 与增量刷新。
- **Causal Replay / Maybe 门禁**：用同一组未来 token 对基线分支和候选分支做受控评分，比较 NLL gain、切换边际和 KL；只有通过门禁的候选才可安排下一轮恢复。
- **Execution Capsule**：保存跨轮次的粗粒度执行状态，供长任务恢复使用。
- **工具调用与 Responses 语义**：保持 Qwen thinking、结构化 tool call、SSE/Responses 事件和取消传播语义。
- **统一资源准入**：在 scheduler 原生 job 之前估算 KV、请求槽位、Mamba 槽位和临时 logprob workspace，并执行跨 TP rank 的原子准入。
- **可观测性控制台**：提供健康状态、Q×K 候选、Judge 决策、Native restore、Self-Ask、Causal Replay、Maybe 和原始遥测查看。

## 使用的核心技术

| 技术 | 用途 |
|---|---|
| **SGLang** | 推理调度、Continuous Batching、Radix Cache、内部 scheduler job 和 OpenAI-compatible HTTP 服务 |
| **PyTorch / CUDA / NCCL** | TP=2 执行、GPU 状态、跨 rank 通信和模型前向 |
| **Qwen Hybrid Attention** | Full-Attention 层负责 KV；Gated DeltaNet 层负责 recurrent/conv state |
| **Tensor Parallelism** | 两张 GPU 上的真正模型并行，启动参数为 `--tp-size 2` |
| **FP8 KV Cache / BF16 State** | 降低 KV 显存占用，同时保留 BF16 模型状态正确性基线 |
| **Tensor Bank** | 持久化文档级原生 K/V、显著 token 位置和完整 GDN state |
| **Raw Attention-Q × K** | 使用模型原生 Attention 信号完成候选检索，不依赖外部 Embedding 服务 |
| **16-token Local Window** | 要求候选在连续局部窗口内得到支持，降低孤立 token 极值误召回 |
| **Median/MAD Relative Evidence** | 记录 per-query 稳健背景分数、相对分数和 margin；默认作为 Shadow 审计数据 |
| **Reference Judge** | 使用受约束 JSON 决策做最终语义准入 |
| **Scheduler-native Internal Jobs** | Judge、Self-Ask、Self-Answer、Capsule 和 Replay 不递归调用 HTTP |
| **Observer / Self-Ask** | decode 中检测持续不确定性并请求内部问题 |
| **Causal Replay / Maybe** | 用共同未来 token 做基线/候选反事实比较，失败关闭，不修改已经输出的 token |
| **FastAPI / Uvicorn** | QWEN-EXO 控制面、健康检查、知识库管理和遥测 API |
| **React / Vite / Tailwind** | 中文操作控制台和召回轨迹可视化 |
| **Docker** | 固定 SGLang/CUDA/Torch 运行基线，隔离模型服务环境 |

## 环境要求

默认验证配置：

- Linux x86_64
- Docker + NVIDIA Container Toolkit
- 2 × NVIDIA RTX 4090，单卡约 48 GiB
- NVIDIA Driver 550.78 或部署验证过的兼容版本
- CUDA 12.6 基础镜像
- Qwen Hybrid Dense 27B 或 MoE 35B-A3B 结构
- 运行时服务：`127.0.0.1:30000`
- 服务模型 ID：`duckgpt`

兼容性由 checkpoint 的 `config.json` 结构判断，不由目录名、营销版本号或容器别名判断。启动会拒绝不支持的模型结构。

## 安装与启动

### 1. 获取代码

```bash
git clone https://github.com/huoji120/QWEN-EXO-booster.git
cd QWEN-EXO-booster
```

如果使用内部仓库或已有 checkout，直接进入项目目录即可。

### 2. 设置模型和运行数据目录

运行数据必须放在 checkout 之外。不要把模型权重、Tensor Bank、遥测、请求轨迹和训练产物放进 Git 工作区。

```bash
export QWEN_EXO_MODEL_PATH=/data/models/Qwen3.5-27B
export QWEN_EXO_DATA_PATH=/data/qwen-exo-runtime
export QWEN_EXO_IMAGE=qwen-exo-booster:sglang-v0.5.16-driver550

mkdir -p "$QWEN_EXO_DATA_PATH"
```

### 3. 构建镜像并执行预检

```bash
bash scripts/qwen_exo/build_image.sh
```

构建脚本会执行：

1. CUDA 和 GPU 检查；
2. SGLang 与 QWEN-EXO 模块导入检查；
3. GPU kernel 检查；
4. 固定基础镜像和当前 Git revision 的 Docker 镜像构建。

### 4. 启动服务

```bash
bash scripts/qwen_exo/launch_js4090.sh
```

默认关键参数：

```text
TP=2
weights dtype=BF16
quantization=FP8
KV cache=FP8 E4M3
context length=102400
page size=64
observer=active
adaptive refresh=enabled
```

启动脚本在 Docker 启动前检查 GPU 是否已有 compute PID。不要在同一组 GPU 上同时运行两个推理后端。

### 5. 等待就绪

```bash
curl -f http://127.0.0.1:30000/qwen-exo/health
curl -s http://127.0.0.1:30000/qwen-exo/status
curl -s http://127.0.0.1:30000/v1/models
```

只有当 `/qwen-exo/health` 返回以下状态，服务才算可用：

```json
{
  "status": "ok",
  "runtime_state": "ready"
}
```

## 访问中文控制台

控制台默认只绑定在 `127.0.0.1`，不应该直接暴露到公网。远程访问使用 SSH 本地端口转发。

### 在 GPU 服务器上确认服务

```bash
ssh <gpu-host> 'curl -f http://127.0.0.1:30000/qwen-exo/health'
```

### 在本地建立 SSH 隧道

```bash
ssh -N -L 30000:127.0.0.1:30000 <gpu-user>@<gpu-host>
```

保持该终端运行，然后在本地浏览器打开：

```text
http://127.0.0.1:30000/qwen-exo/
```

### 控制台页面

- `/qwen-exo/`：中文用户工作区、对话和普通操作入口；
- `/qwen-exo/admin`：运维控制台；
- 页面中的“召回轨迹”：查看请求候选、Q×K、Semantic Judge、Native restore、Self-Ask、Causal Replay、Maybe 和原始事件；
- `/qwen-exo/recall-trace`：兼容的召回轨迹页面。

控制台可以查看和管理：

- Knowledge Markdown；
- PolicyData；
- Tensor Bank 编译状态；
- 服务配置和 revision 健康状态；
- 请求遥测与 Recall Trace；
- Reflection Memory 任务；
- Observer、Adaptive Refresh、Score Bias 和 Causal Replay 状态。

Knowledge 和 PolicyData 的内容修改属于 loopback 控制面。请只通过 SSH 隧道或可信的运维反向代理访问，不要直接绑定公网网卡。

## API 快速示例

### Responses 推理

```bash
curl --no-buffer http://127.0.0.1:30000/v1/responses \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "duckgpt",
    "input": "解释当前服务的混合注意力状态如何恢复。",
    "stream": true,
    "max_output_tokens": 256
  }'
```

### 查询知识库元数据

```bash
curl http://127.0.0.1:30000/qwen-exo/knowledge
```

### 查询召回轨迹

```bash
curl 'http://127.0.0.1:30000/qwen-exo/recall-trace?limit=10'
```

### 查询遥测

```bash
curl 'http://127.0.0.1:30000/qwen-exo/telemetry?limit=100'
```

默认遥测会脱敏：prompt、输出、reasoning、工具参数、参考文档和密钥不会直接写入遥测。需要详细 API 字段时，阅读 [API 与控制台说明](docs/qwen_exo/API.md)。

## 本地验证

不加载线上模型时，可以运行 Python 回归：

```bash
PYTHONPATH=python python -m pytest test/registered/qwen_exo -q
```

构建控制台：

```bash
cd frontend/qwen-exo
npm ci
npm run build
```

GPU 部署验证：

```bash
python3 scripts/qwen_exo/check_cuda.py
python3 scripts/qwen_exo/check_imports.py
python3 scripts/qwen_exo/check_kernels.py
python3 scripts/qwen_exo/smoke_contracts.py
```

## 目录说明

```text
python/qwen_exo_booster/       QWEN-EXO runtime、Memory Pipeline、Judge、Observer、API
python/sglang/                 SGLang 二开代码和模型/scheduler 集成
scripts/qwen_exo/              构建、启动、预检、smoke、评测和 Bank 工具
docker/                        QWEN-EXO Dockerfile 和部署配置
frontend/qwen-exo/             React/Vite 中文控制台
docs/qwen_exo/                 架构、API、部署和验证文档
test/registered/qwen_exo/      注册回归测试
scripts/qwen_exo/corpus/       版本化 Knowledge、PolicyData 和可选 Cognition 源文件
```

## 重要安全边界

- 不要把模型权重、Tensor Bank、运行遥测、请求轨迹、训练数据或编辑器权重提交到 Git；
- 控制台默认是 loopback-only，不要直接暴露 `/qwen-exo/knowledge`、`/qwen-exo/policydata` 等写接口；
- `causal_replay` 只比较候选分支，不会改写已经输出的 token；
- Judge、Native state binding 和 resource admission 都失败关闭；
- 项目不调用外部 LLM，不执行隐式外部学习；
- 不要根据 Qwen checkpoint 目录名称判断兼容性；
- 不要在同一 GPU 上同时启动旧后端和 QWEN-EXO。

## 深入阅读

1. [架构与状态契约](docs/qwen_exo/ARCHITECTURE.md)
2. [双 RTX 4090 部署指南](docs/qwen_exo/SERVER_27B_DEPLOYMENT.md)
3. [API、遥测、安全与控制台](docs/qwen_exo/API.md)
4. [实现进度与验证证据](docs/qwen_exo/IMPLEMENTATION_PROGRESS.md)
5. [Demo 到生产运行时迁移矩阵](docs/qwen_exo/DEMO_MIGRATION_MATRIX.md)

## 当前定位

QWEN-EXO-booster 是一个围绕 SGLang 的模型运行时二开项目。它重点解决的是 **Qwen Hybrid 模型的原生状态管理、长上下文记忆召回、内部任务调度、Agent 连续性和可验证运维**。任何“模型能力提升”或“准确率提升”结论，都必须由固定模型、固定上下文、固定并发和固定输出长度的对照评测证明，不能只根据单次演示推断。
