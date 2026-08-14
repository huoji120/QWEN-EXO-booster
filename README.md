# QWEN-EXO-booster

<p align="center">
  <strong>简体中文</strong> · <a href="README_EN.md">English</a>
</p>

![img](/banner.png)
基于 **SGLang 二次开发** 的 Qwen 混合注意力推理后端。极大的增强QWEN系列模型能力。
> 支持MAC和LINUX，windows系列sglang不支持建议在WSL中使用,支持QWEN3.5到QWEN3.8的所有模型包括二次修改模型,支持MOE模型.我们推荐使用QWEN3.8-27B这个模型.

## 为什么是QWEN-EXO
### 原生知识库注入
不同于RAG,我们开发了一种基于模型注意力的知识召回机制,这种机制能让模型精准的回忆自己需要的知识而不依赖任何RAG架构.极大的补充了QEWN系列模型知识面不广的情况。
增加模型知识变得非常简单,不需要微调,你只需要给他灌注知识库就行.重要的是,你不需要微调,只需要写知识库文档即可增强模型能力.
![img](/images/1.png)
### 反思记忆
我们开发了一套基于推理服务端的反思记忆,每次失败或者成功都会进行反思回忆.让模型越来越聪明
![img](/images/2.png)
### 可观测性
你可以审查模型注入的知识回忆,防止他跑偏
![img](/images/3.png)

## 一句话安装

把本仓库 README 交给大模型，让它在一台装有 **Docker、NVIDIA Container Toolkit、两张 RTX 4090、NVIDIA 驱动 550+** 的 Linux 主机上执行：

```text
请阅读这个仓库的 README.md 和 docs/qwen_exo/SERVER_27B_DEPLOYMENT.md，确认 QWEN_EXO_MODEL_PATH 指向兼容的 Qwen Hybrid checkpoint、QWEN_EXO_DATA_PATH 指向独立运行数据目录，然后执行 bash scripts/qwen_exo/build_image.sh 和 bash scripts/qwen_exo/launch_js4090.sh；启动后等待 http://127.0.0.1:30000/qwen-exo/health 返回 runtime_state=ready，再通过 SSH 隧道访问控制台。
```

Apple Silicon Mac 使用仓库内的原生 MLX 执行链路，不需要 Docker 或 CUDA：

```bash
bash scripts/qwen_exo/install_mlx.sh
export QWEN_EXO_MODEL_PATH=/path/to/Qwen3.8-27B
export QWEN_EXO_DATA_PATH=/path/to/qwen-exo-runtime
bash scripts/qwen_exo/launch_mlx.sh
```

完整依赖、固定后端参数和验证边界见 [Apple Silicon MLX 部署指南](docs/qwen_exo/APPLE_SILICON_MLX_DEPLOYMENT.md)。

## 安装与启动

### 打开任意agent终端
输入 帮我部署这个sglang二开的QWEN-EXO项目:
```bash
git clone https://github.com/huoji120/QWEN-EXO-booster.git
cd QWEN-EXO-booster
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

Apple Silicon MLX 验证：

```bash
.venv/bin/python scripts/qwen_exo/check_mlx.py
PYTHONPATH=python .venv/bin/python -m pytest \
  test/registered/qwen_exo/test_mlx_preflight.py \
  test/registered/qwen_exo/test_mlx_launcher.py \
  test/registered/qwen_exo/test_hybrid_state.py \
  test/registered/qwen_exo/test_config_runtime.py -q
```

## 目录说明

```text
python/qwen_exo_booster/       QWEN-EXO runtime、Memory Pipeline、Judge、Observer、API
python/sglang/                 SGLang 二开代码和模型/scheduler 集成
scripts/qwen_exo/              构建、启动、预检、smoke 和评测工具
scripts/qwen_exo/corpus/knowledge/  统一知识预编译源（事实知识与 reflection-memory）
scripts/qwen_exo/corpus/policydata/  版本化 PolicyData 源文件
scripts/qwen_exo/corpus/cognition/   可选 Cognition 源文件
docker/                        QWEN-EXO Dockerfile 和部署配置
frontend/qwen-exo/             React/Vite 中文控制台
docs/qwen_exo/                 架构、API、部署和验证文档
test/registered/qwen_exo/      注册回归测试

```

运行目录中的 Markdown/JSON 源不按模型复制。模型切换只切换权重路径和 `model-profiles/<模型指纹>/state-*`；目标模型从统一预编译源重新生成自己的 Tensor Bank 和 Native Bank，因而不同模型可使用不同分词、K/V 与 GDN 状态而不会产生内容分叉。生成的 Bank 只属于本地模型、量化与拓扑，不是可移植发布物。


模型目录中的 `checkpoint_quantization`、`runtime_quantization` 与运行时 `kv_cache_dtype` 必须分开读取：27B GPTQ 是 W4A16，FP8 只用于 Full-Attention KV cache；GDN/Mamba recurrent/conv state 仍由运行时单独管理，不能把 `--quantization fp8` 当作 GPTQ 叠加压缩。


## 重要安全边界

## 深入阅读

1. [架构与状态契约](docs/qwen_exo/ARCHITECTURE.md)
2. [双 RTX 4090 部署指南](docs/qwen_exo/SERVER_27B_DEPLOYMENT.md)
3. [Apple Silicon MLX 部署指南](docs/qwen_exo/APPLE_SILICON_MLX_DEPLOYMENT.md)
4. [API、遥测、安全与控制台](docs/qwen_exo/API.md)
5. [实现进度与验证证据](docs/qwen_exo/IMPLEMENTATION_PROGRESS.md)
6. [Demo 到生产运行时迁移矩阵](docs/qwen_exo/DEMO_MIGRATION_MATRIX.md)

## 当前定位

QWEN-EXO-booster 是一个围绕 SGLang 的模型运行时二开项目。它重点解决的是 **Qwen Hybrid 模型的原生状态管理、长上下文记忆召回、内部任务调度、Agent 连续性和可验证运维**。任何“模型能力提升”或“准确率提升”结论，都必须由固定模型、固定上下文、固定并发和固定输出长度的对照评测证明，不能只根据单次演示推断。
