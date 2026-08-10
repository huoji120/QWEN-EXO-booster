from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from qwen_exo_booster.config import QwenExoConfig, QwenExoFeatureFlags

_SERVICE_CONFIG_SCHEMA = 1
_DEFAULT_CONFIG_PATH = Path("/data/qwen-exo-booster/service-config.json")


class ServiceConfigError(ValueError):
    def __init__(self, code: str, message: str, *, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field

    def public_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": str(self)}
        if self.field is not None:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True, slots=True)
class ServiceSetting:
    key: str
    flag: str
    group: str
    label: str
    description: str
    value_type: str
    default: bool | int | float | str
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    choices: tuple[str, ...] = ()
    choice_labels: dict[str, str] | None = None
    unit: str | None = None
    restart_required: bool = True

    @property
    def negative_flag(self) -> str:
        return "--no-" + self.flag.removeprefix("--")

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "group": self.group,
            "label": self.label,
            "description": self.description,
            "type": self.value_type,
            "default": self.default,
            "restart_required": self.restart_required,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.step is not None:
            payload["step"] = self.step
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.choice_labels:
            payload["choice_labels"] = dict(self.choice_labels)
        if self.unit is not None:
            payload["unit"] = self.unit
        return payload


_GROUPS = (
    {
        "id": "capacity",
        "label": "容量与调度",
        "description": "上下文窗口、静态显存预留、运行并发与预填充批量预算；决定吞吐、首字延迟和调度拥塞边界。",
    },
    {
        "id": "memory",
        "label": "知识检索与准入",
        "description": "定义 Attention-Q × K 召回、语义准入及原生状态注入的门限与预算；控制候选产生到实际进入上下文的完整链路。",
    },
    {
        "id": "tensor_bank",
        "label": "Tensor Bank 编译",
        "description": "控制文档编译为原生 Full-Attention K/V 与 GDN 状态的长度上限、显著片段提取及片段半径。",
    },
    {
        "id": "observer",
        "label": "运行观测",
        "description": "解码期间持续监测局部惊奇度与 Attention-Q 漂移；触发条件决定 Self-Ask、候选刷新与回放激活频率。",
    },
    {
        "id": "replay",
        "label": "因果回放",
        "description": "以同一未来 token 窗口对基线分支与候选分支评分；仅在损失增益达标且分布差异受控时采纳。",
    },
    {
        "id": "score_bias",
        "label": "注意力控制",
        "description": "依据当前 Attention-Q 对历史轨迹片段施加有界注意力加权；作用范围限于注意力分数，不修改权重或请求文本。",
    },
    {
        "id": "privacy",
        "label": "遥测与控制台",
        "description": "控制遥测原文记录范围与召回轨迹控制台的默认显示范围。",
    },
    {
        "id": "post_tool_evidence",
        "label": "工具后证据",
        "description": "当外部知识均未通过准入时，仅使用最新工具结果中的可验证事实补全当前请求。",
    },
    {
        "id": "context_integrity",
        "label": "上下文完整性",
        "description": "由模型将最新工具内容与最近会话上下文进行语义对照；预算自动取模型最大上下文的配置分之一。",
    },
    {
        "id": "reflection_memory",
        "label": "反思记忆",
        "description": "在工具轨迹空闲后提炼可复用经验，热写入知识库并重建原生 Tensor Bank。",
    },
    {
        "id": "compaction",
        "label": "上下文压缩",
        "description": "控制 Responses 历史压缩的摘要预算与状态复用；只影响显式压缩请求。",
    },
    {
        "id": "editor",
        "label": "轨迹编辑器",
        "description": "在指定模型层施加低秩激活修正，使行为贴近已训练轨迹；强度档位决定修正幅度与结构失真风险。",
    },
)


def _setting(
    key: str,
    group: str,
    label: str,
    description: str,
    value_type: str,
    default: bool | int | float | str,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    choices: tuple[str, ...] = (),
    choice_labels: dict[str, str] | None = None,
    unit: str | None = None,
) -> ServiceSetting:
    return ServiceSetting(
        key=key,
        flag="--" + key.replace("_", "-"),
        group=group,
        label=label,
        description=description,
        value_type=value_type,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        choices=choices,
        choice_labels=choice_labels,
        unit=unit,
    )


SERVICE_SETTINGS = (
    _setting(
        "context_length",
        "capacity",
        "上下文长度",
        "作用：限制单个请求可使用的最大上下文长度。用途：防止超长请求挤占显存；调大可支持更长对话。推荐值：102400（本机双卡实测稳定）。",
        "integer",
        102400,
        minimum=4096,
        maximum=262144,
        step=1024,
        unit="tokens",
    ),
    _setting(
        "mem_fraction_static",
        "capacity",
        "静态显存比例",
        "作用：模型权重和缓存预留的显存比例。用途：调高提升吞吐但更容易溢出，调低更稳。推荐值：0.8。",
        "number",
        0.8,
        minimum=0.1,
        maximum=0.95,
        step=0.01,
    ),
    _setting(
        "max_running_requests",
        "capacity",
        "最大并发请求",
        "作用：调度器同时处理的请求数上限。用途：限制并发防止拥塞排队。推荐值：64；单人使用时 1～8 响应更快。",
        "integer",
        64,
        minimum=1,
        maximum=512,
        step=1,
    ),
    _setting(
        "max_prefill_tokens",
        "capacity",
        "预填充批量上限",
        "作用：一个调度批次可处理的输入 token 总量。用途：影响首字延迟与吞吐的平衡。推荐值：65536。",
        "integer",
        65536,
        minimum=1024,
        maximum=262144,
        step=1024,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_max_internal_fanout",
        "capacity",
        "内部任务并发",
        "作用：语义裁判、自我提问等后台任务的并发上限。用途：限制内部推理对主请求的干扰。推荐值：32。",
        "integer",
        32,
        minimum=1,
        maximum=128,
        step=1,
    ),
    _setting(
        "qwen_exo_max_internal_tokens",
        "capacity",
        "内部任务预算",
        "作用：每个请求允许内部任务消耗的 token 上限。反思记忆启用 Q×K 时先为查询探针预留 1 token，默认最多 3 次生成各使用 4095 token，累计仍不超过 12288。",
        "integer",
        12288,
        minimum=128,
        maximum=65536,
        step=64,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_max_output_tokens",
        "capacity",
        "单轮输出上限",
        "作用：限制一次主请求的思考、回答和工具调用总 token 数；不关闭任何 QWEN-EXO 能力。推荐值：8192。",
        "integer",
        8192,
        minimum=512,
        maximum=32768,
        step=512,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_max_reasoning_tokens",
        "capacity",
        "推理思考上限",
        "作用：模型思考阶段允许的最大 token 数。用途：太短答案可能被截断，太长增加延迟。推荐值：3072。",
        "integer",
        3072,
        minimum=128,
        maximum=32768,
        step=128,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_enable_hybrid_prefix",
        "memory",
        "混合前缀状态",
        "作用：同时管理完整注意力层的键值缓存与线性注意力层的循环状态。用途：关闭后原生记忆恢复不可用。推荐值：开启。",
        "boolean",
        True,
    ),
    _setting(
        "qwen_exo_enable_external_memory",
        "memory",
        "知识库记忆",
        "作用：允许请求检索并恢复知识库文档的状态。用途：关闭后回答只依赖当前对话内容。推荐值：开启。",
        "boolean",
        True,
    ),
    _setting(
        "qwen_exo_enable_policy_data",
        "memory",
        "策略数据",
        "作用：启用执行策略文档的原生状态条件化。用途：让模型遵守既定的工程策略。推荐值：开启。",
        "boolean",
        True,
    ),
    _setting(
        "qwen_exo_enable_reference_judge",
        "memory",
        "语义裁判",
        "作用：对候选记忆做语义相关性判定。用途：防止无关内容进入上下文。推荐值：开启。",
        "boolean",
        True,
    ),
    _setting(
        "qwen_exo_enable_capsule",
        "memory",
        "执行胶囊",
        "作用：保存跨轮次的粗粒度执行状态。用途：长任务中断后恢复进度。推荐值：开启。",
        "boolean",
        True,
    ),
    _setting(
        "qwen_exo_max_candidates",
        "memory",
        "最大候选数",
        "作用：每次检索送入准入流程的候选数量上限。用途：候选越多召回越全，但裁判耗时越长。推荐值：8。",
        "integer",
        8,
        minimum=1,
        maximum=64,
        step=1,
    ),
    _setting(
        "qwen_exo_qk_recall_preset",
        "memory",
        "Q×K 召回严格度",
        "工作原理：控制 Attention-Q 与文档 K 相似度的第一道门槛，并同时约束领先文档与第二名的差距。宽松会交给语义审计更多候选；标准保持当前行为；严格会提前过滤弱相关或难分胜负的文档。推荐：标准。",
        "string",
        "balanced",
        choices=("broad", "balanced", "strict"),
        choice_labels={
            "broad": "高召回：尽量不漏，交给审计模型过滤",
            "balanced": "标准：保持召回与误命中的平衡",
            "strict": "高精度：只保留高分且领先明确的文档",
        },
    ),
    _setting(
        "qwen_exo_max_memory_tokens",
        "memory",
        "知识库预算",
        "作用：单次请求允许恢复的知识库内容上限。用途：防止记忆内容挤占对话空间。推荐值：8192。",
        "integer",
        8192,
        minimum=128,
        maximum=32768,
        step=128,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_max_policy_tokens",
        "memory",
        "策略预算",
        "作用：单个策略文档允许恢复的内容上限。推荐值：4096。",
        "integer",
        4096,
        minimum=128,
        maximum=16384,
        step=128,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_qk_expansion_margin",
        "memory",
        "检索扩张阈值",
        "作用：候选进入语义裁判前所需的最小领先幅度。用途：过滤不自信的检索结果。推荐值：0.01。",
        "number",
        0.01,
        minimum=0,
        maximum=1,
        step=0.001,
    ),
    _setting(
        "qwen_exo_qk_prefilter_mode",
        "memory",
        "裁判前预过滤",
        "作用：最高检索分低于门槛且没有跨轮恢复证据时跳过裁判；多个高分候选分差较小时改由大模型并排比较并只选一个。关闭=始终审计；启用=过滤弱候选并比较歧义候选。推荐值：启用。",
        "string",
        "active",
        choices=("off", "active"),
        choice_labels={"off": "关闭", "active": "启用"},
    ),
    _setting(
        "qwen_exo_qk_max_candidates_per_document",
        "memory",
        "单文档候选上限",
        "作用：同一文档最多允许进入语义裁判的不同页面候选数，最佳页面始终优先保留。用途：避免单个文档占满候选名额。推荐值：1。",
        "integer",
        1,
        minimum=1,
        maximum=4,
        step=1,
    ),
    _setting(
        "qwen_exo_qk_only_knowledge",
        "memory",
        "仅注意力检索",
        "作用：跳过词法匹配，只用注意力查询信号路由记忆。用途：开启后更精确但可能漏召回。推荐值：关闭。",
        "boolean",
        False,
    ),
    _setting(
        "qwen_exo_tensor_bank_max_document_tokens",
        "tensor_bank",
        "文档长度上限",
        "作用：单个文档允许编译入库的最大 token 数；不得超过当前上下文长度减 2048。当前 102400 上下文推荐值：100352。",
        "integer",
        100352,
        minimum=64,
        maximum=131072,
        step=64,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_tensor_bank_salient_token_budget",
        "tensor_bank",
        "显著片段预算",
        "作用：单个文档预留的精确 Full-Attention K/V token 成本。当前固定预算：4096。",
        "integer",
        4096,
        minimum=64,
        maximum=32768,
        step=64,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_tensor_bank_surprisal_threshold",
        "tensor_bank",
        "惊奇度阈值",
        "作用：选取高惊奇 token 的门槛。用途：调低保留更多片段；智能体轨迹内容惊奇度普遍偏高，需要调高。推荐值：6.0（普通文档）；30（长轨迹）。",
        "number",
        6.0,
        minimum=0,
        maximum=32,
        step=0.1,
    ),
    _setting(
        "qwen_exo_tensor_bank_span_tokens",
        "tensor_bank",
        "片段半径",
        "作用：每个高惊奇 token 周围保留的上下文长度。用途：半径越大上下文越完整，但片段合并后容易超预算。推荐值：16。",
        "integer",
        16,
        minimum=1,
        maximum=2048,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_observer_mode",
        "observer",
        "观测器模式",
        "作用：解码期观测的工作模式。关闭=完全不观测；仅观测=只记录不干预；主动干预=允许触发自我提问与回放。推荐值：主动干预。",
        "string",
        "active",
        choices=("off", "shadow", "active"),
        choice_labels={"off": "关闭", "shadow": "仅观测", "active": "主动干预"},
    ),
    _setting(
        "qwen_exo_enable_adaptive_refresh",
        "observer",
        "自适应刷新",
        "作用：持续不确定时自动触发自我提问与候选刷新。用途：让模型在推理中主动补充记忆。推荐值：开启。",
        "boolean",
        True,
    ),
    _setting(
        "qwen_exo_immediate_uncertainty_retrieval",
        "observer",
        "即时自我提问",
        "作用：观测命中时立即启动隐藏的自我提问。用途：缩短不确定状态的持续时间。推荐值：开启。",
        "boolean",
        True,
    ),
    _setting(
        "qwen_exo_observer_surprisal_threshold",
        "observer",
        "局部惊奇度阈值",
        "作用：选中 token 的局部平均惊奇度下限。用途：调低更敏感、触发更多；调高更安静。推荐值：0.8。",
        "number",
        0.8,
        minimum=0,
        maximum=20,
        step=0.05,
    ),
    _setting(
        "qwen_exo_observer_surprisal_window",
        "observer",
        "惊奇度窗口",
        "作用：计算局部惊奇度的 token 窗口大小。推荐值：8。",
        "integer",
        8,
        minimum=2,
        maximum=128,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_observer_surprisal_margin",
        "observer",
        "惊奇度增量",
        "作用：局部窗口相对历史窗口的最小增长幅度。用途：防止缓慢波动的误触发。推荐值：0.2。",
        "number",
        0.2,
        minimum=0,
        maximum=10,
        step=0.05,
    ),
    _setting(
        "qwen_exo_observer_q_drift_threshold",
        "observer",
        "查询漂移阈值",
        "作用：注意力查询向量漂移触发观测的门槛。推荐值：0.35。",
        "number",
        0.35,
        minimum=0,
        maximum=2,
        step=0.01,
    ),
    _setting(
        "qwen_exo_observer_cooldown_tokens",
        "observer",
        "触发冷却间隔",
        "作用：同一请求两次观测触发的最小 token 间隔。推荐值：64。",
        "integer",
        64,
        minimum=1,
        maximum=4096,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_observer_max_triggers",
        "observer",
        "单请求触发上限",
        "作用：每个请求最多允许的刷新次数（当前实现最多 1 次）。推荐值：1。",
        "integer",
        1,
        minimum=0,
        maximum=1,
        step=1,
    ),
    _setting(
        "qwen_exo_observer_q_pre_tokens",
        "observer",
        "触发前查询窗口",
        "作用：触发点之前保留的注意力查询快照数。推荐值：8。",
        "integer",
        8,
        minimum=1,
        maximum=128,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_observer_q_post_tokens",
        "observer",
        "触发后查询窗口",
        "作用：触发点之后保留的注意力查询快照数。推荐值：4。",
        "integer",
        4,
        minimum=1,
        maximum=128,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_observer_recovery_tokens",
        "observer",
        "恢复判定窗口",
        "作用：判断不确定性是否恢复所需观测的未来 token 数。推荐值：8。",
        "integer",
        8,
        minimum=1,
        maximum=128,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_replay_observation_tokens",
        "replay",
        "观测窗口",
        "作用：各回放分支共同评分的真实未来 token 数。推荐值：8。",
        "integer",
        8,
        minimum=2,
        maximum=128,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_replay_prefix_tokens",
        "replay",
        "父前缀长度",
        "作用：回放分支复用的父请求前缀长度。推荐值：1024。",
        "integer",
        1024,
        minimum=1,
        maximum=16384,
        step=128,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_replay_max_candidates",
        "replay",
        "回放候选数",
        "作用：一次因果回放允许比较的候选分支数。推荐值：2。",
        "integer",
        2,
        minimum=1,
        maximum=8,
        step=1,
    ),
    _setting(
        "qwen_exo_replay_reference_tokens",
        "replay",
        "参考内容预算",
        "作用：一个回放候选注入的参考内容 token 上限。推荐值：128。",
        "integer",
        128,
        minimum=1,
        maximum=4096,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_replay_minimum_gain",
        "replay",
        "最小增益",
        "作用：候选分支优于基线所需的最低损失增益。推荐值：0.02。",
        "number",
        0.02,
        minimum=0,
        maximum=10,
        step=0.005,
    ),
    _setting(
        "qwen_exo_replay_switch_margin",
        "replay",
        "切换增益",
        "作用：替换当前候选所需的额外损失增益。推荐值：0.05。",
        "number",
        0.05,
        minimum=0,
        maximum=10,
        step=0.005,
    ),
    _setting(
        "qwen_exo_replay_maybe_kl_cap",
        "replay",
        "分布差异上限",
        "作用：试探性准入接受的选中 token 分布差异上限。推荐值：4.0。",
        "number",
        4.0,
        minimum=0,
        maximum=100,
        step=0.1,
    ),
    _setting(
        "qwen_exo_score_bias_mode",
        "score_bias",
        "运行模式",
        "作用：历史轨迹对注意力打分的介入方式。关闭=完全停用；仅打分=只记录不干预；施加偏置=对选中轨迹块施加有界注意力偏置。推荐值：施加偏置。",
        "string",
        "trajectory_active",
        choices=("off", "trajectory_shadow", "trajectory_active"),
        choice_labels={
            "off": "关闭",
            "trajectory_shadow": "仅打分",
            "trajectory_active": "施加偏置",
        },
    ),
    _setting(
        "qwen_exo_score_bias_min_surprisal",
        "score_bias",
        "最小平均惊奇度",
        "作用：轨迹块参与偏置的惊奇度下限。推荐值：0.8。",
        "number",
        0.8,
        minimum=0,
        maximum=20,
        step=0.05,
    ),
    _setting(
        "qwen_exo_score_bias_max",
        "score_bias",
        "最大偏置",
        "作用：施加到注意力分数上的绝对上限。用途：防止偏置压过正常注意力。推荐值：0.25。",
        "number",
        0.25,
        minimum=0,
        maximum=1,
        step=0.01,
    ),
    _setting(
        "qwen_exo_score_bias_half_life_steps",
        "score_bias",
        "半衰期",
        "作用：历史轨迹块权重随轮次衰减的半衰期。推荐值：4。",
        "number",
        4.0,
        minimum=0.1,
        maximum=128,
        step=0.5,
        unit="turns",
    ),
    _setting(
        "qwen_exo_score_bias_max_blocks",
        "score_bias",
        "历史块上限",
        "作用：进入计算核的历史轨迹块数量上限。推荐值：8。",
        "integer",
        8,
        minimum=1,
        maximum=32,
        step=1,
    ),
    _setting(
        "qwen_exo_score_bias_min_age_steps",
        "score_bias",
        "最小轮次年龄",
        "作用：轨迹块参与偏置前必须经过的轮次数。推荐值：2。",
        "integer",
        2,
        minimum=1,
        maximum=256,
        step=1,
        unit="turns",
    ),
    _setting(
        "qwen_exo_score_bias_max_age_steps",
        "score_bias",
        "最大轮次年龄",
        "作用：轨迹块参与偏置的最大轮次年龄。推荐值：16。",
        "integer",
        16,
        minimum=1,
        maximum=512,
        step=1,
        unit="turns",
    ),
    _setting(
        "qwen_exo_score_bias_tail_tokens",
        "score_bias",
        "尾部长度上限",
        "作用：轨迹尾部可被切成候选块的 token 数。推荐值：4096。",
        "integer",
        4096,
        minimum=0,
        maximum=32768,
        step=128,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_score_bias_tail_ratio",
        "score_bias",
        "尾部比例",
        "作用：从轨迹尾部纳入候选的比例。推荐值：0.15。",
        "number",
        0.15,
        minimum=0,
        maximum=0.99,
        step=0.01,
    ),
    _setting(
        "qwen_exo_score_bias_selected_blocks",
        "score_bias",
        "选中块数",
        "作用：一次请求实际施加偏置的最高相关块数。推荐值：2。",
        "integer",
        2,
        minimum=1,
        maximum=32,
        step=1,
    ),
    _setting(
        "qwen_exo_score_bias_query_window",
        "score_bias",
        "查询窗口",
        "作用：计算轨迹相关性使用的注意力查询窗口。推荐值：8。",
        "integer",
        8,
        minimum=1,
        maximum=128,
        step=1,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_score_bias_min_relevance",
        "score_bias",
        "最小相关性",
        "作用：轨迹块相关性余弦分数下限。推荐值：0.0。",
        "number",
        0.0,
        minimum=-1,
        maximum=1,
        step=0.01,
    ),
    _setting(
        "qwen_exo_score_bias_relevance_margin",
        "score_bias",
        "相关性边距",
        "作用：最佳块超过其余块所需的最小分数差。推荐值：0.005。",
        "number",
        0.005,
        minimum=0,
        maximum=1,
        step=0.001,
    ),
    _setting(
        "qwen_exo_score_bias_anchor_bias",
        "score_bias",
        "系统锚点偏置",
        "作用：对原始 system instructions 的少量 token span 提供有界保护，避免轨迹偏置造成约束漂移。默认关闭；建议先用 0.01 做对照。",
        "number",
        0.0,
        minimum=0,
        maximum=0.05,
        step=0.001,
    ),
    _setting(
        "qwen_exo_score_bias_anchor_max_blocks",
        "score_bias",
        "系统锚点块数",
        "作用：系统 instructions 参与 decode 锚定的最多 128-token 块数。",
        "integer",
        2,
        minimum=1,
        maximum=4,
        step=1,
    ),
    _setting(
        "qwen_exo_telemetry_text_mode",
        "privacy",
        "遥测原文模式",
        "作用：遥测中原文的记录策略。全部脱敏=不记录任何原文；仅编辑片段=只记录被轨迹微调影响的请求且每段有界截断；全量记录=记录全部原文（页面可能变慢）。推荐值：仅编辑片段。",
        "string",
        "off",
        choices=("off", "edited", "all"),
        choice_labels={
            "off": "全部脱敏",
            "edited": "仅编辑片段",
            "all": "全量记录",
        },
    ),
    _setting(
        "qwen_exo_console_trace_default_scope",
        "privacy",
        "召回轨迹默认范围",
        "召回轨迹控制台的默认显示范围：记忆活动保留知识/策略注入或 Self-Ask 请求，仅实际召回进一步排除仅 Self-Ask 请求，全部请求包含无召回事件。推荐值：记忆活动。",
        "string",
        "activity",
        choices=("activity", "actual", "all"),
        choice_labels={
            "activity": "记忆活动",
            "actual": "仅实际召回",
            "all": "全部请求",
        },
    ),
    _setting(
        "qwen_exo_context_evidence_mode",
        "post_tool_evidence",
        "工具后上下文证据",
        "作用：外部知识与 PolicyData 均未通过语义准入时，审查最新工具结果；启用时只把明确回答问题的直接观察作为当前请求的临时证据。默认启用。",
        "string",
        "active",
        choices=("off", "active"),
        choice_labels={"off": "关闭", "active": "启用"},
    ),
    _setting(
        "qwen_exo_context_integrity_mode",
        "context_integrity",
        "完整性检查",
        "作用：由模型根据最新工具内容和最近会话上下文审查过期事实；不按工具名称或硬编码规则判断。只在工具原文明确推翻旧结论时写入更正。默认启用。",
        "string",
        "active",
        choices=("off", "active"),
        choice_labels={"off": "关闭", "active": "启用"},
    ),
    _setting(
        "qwen_exo_context_integrity_context_divisor",
        "context_integrity",
        "完整性上下文比例",
        "作用：完整性检查自动使用模型最大上下文的对应分之一；默认 3，即使用三分之一。模型上下文变化时预算同步变化。",
        "integer",
        3,
        minimum=2,
        maximum=8,
        step=1,
    ),
    _setting(
        "qwen_exo_reflection_memory_mode",
        "reflection_memory",
        "反思记忆",
        "作用：工具轨迹空闲达到门限后，先用 Q×K 检索已有反思，再由模型研判是更新同一记忆还是插入新记忆；最终热写入知识库并原地重建 Tensor Bank。聊天和内部任务不会触发。默认启用。",
        "string",
        "active",
        choices=("off", "active"),
        choice_labels={"off": "关闭", "active": "启用"},
    ),
    _setting(
        "qwen_exo_reflection_memory_idle_seconds",
        "reflection_memory",
        "空闲门限",
        "作用：外部工具轨迹在没有新工具结果后等待多久才允许提炼。不是完成信号；只按空闲计时。默认 600 秒，至少 60 秒。",
        "number",
        600.0,
        minimum=60,
        maximum=86400,
        step=60,
        unit="seconds",
    ),
    _setting(
        "qwen_exo_reflection_memory_min_events",
        "reflection_memory",
        "最少工具事件",
        "作用：防止普通聊天或过短轨迹进入反思；必须有至少这么多外部工具结果。内部 Self-Ask、Judge、压缩和 Bank 任务不计入。默认 3。",
        "integer",
        3,
        minimum=2,
        maximum=64,
        step=1,
        unit="events",
    ),
    _setting(
        "qwen_exo_reflection_memory_min_tokens",
        "reflection_memory",
        "最少轨迹 Token",
        "作用：过滤没有足够技术细节的短轨迹。默认 256。",
        "integer",
        256,
        minimum=0,
        maximum=32768,
        step=64,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_reflection_memory_max_attempts",
        "reflection_memory",
        "工具重试次数",
        "作用：反思记忆工具调用格式失败时允许重试，硬上限为 3 次。超过后失败关闭，不写入知识库。默认 3。",
        "integer",
        3,
        minimum=1,
        maximum=3,
        step=1,
        unit="attempts",
    ),
    _setting(
        "qwen_exo_reflection_memory_max_output_tokens",
        "reflection_memory",
        "输出预算",
        "作用：限制一次 think 加反思工具调用的输出预算。默认 4096，范围 512–8192；累计重试预算由内部任务预算覆盖。",
        "integer",
        4096,
        minimum=512,
        maximum=8192,
        step=256,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_reflection_memory_max_history_tokens",
        "reflection_memory",
        "历史预算",
        "作用：反思记忆可读取的轨迹来源总预算。当前服务上下文为 102400，需预留系统提示和 3072 输出，因此默认 92160；超长轨迹按事件去重并保留最近证据，同时记录省略审计。",
        "integer",
        92160,
        minimum=1024,
        maximum=96256,
        step=1024,
        unit="tokens",
    ),
    _setting(
        "qwen_exo_response_compaction_mode",
        "compaction",
        "Responses 上下文压缩",
        "作用：开启后，POST /v1/responses/compact 生成普通文本摘要，并复用上一轮已保存的 DeltaNet 状态与高惊奇度 K/V；关闭后端点拒绝请求。默认启用。",
        "string",
        "active",
        choices=("off", "active"),
        choice_labels={"off": "关闭", "active": "开启"},
    ),
    _setting(
        "qwen_exo_activation_editor_strength",
        "editor",
        "微调强度",
        "作用：选择当前联合轨迹编辑器的作用强度。标准、明显、最强依次增强编辑影响；不自动改档，由用户按实际任务效果选择。",
        "string",
        "2.0",
        choices=("1.0", "2.0", "3.0"),
        choice_labels={
            "1.0": "标准（1×）",
            "2.0": "明显（2×）",
            "3.0": "最强（3×）",
        },
    ),
)

_SETTINGS_BY_KEY = {setting.key: setting for setting in SERVICE_SETTINGS}

_SHADOW_TO_ACTIVE_KEYS = frozenset(
    {
        "qwen_exo_qk_prefilter_mode",
        "qwen_exo_context_evidence_mode",
        "qwen_exo_context_integrity_mode",
    }
)

_LEGACY_REFLECTION_MEMORY_KEYS = {
    "qwen_exo_dream_reflection_mode": "qwen_exo_reflection_memory_mode",
    "qwen_exo_dream_reflection_idle_seconds": "qwen_exo_reflection_memory_idle_seconds",
    "qwen_exo_dream_reflection_min_events": "qwen_exo_reflection_memory_min_events",
    "qwen_exo_dream_reflection_min_tokens": "qwen_exo_reflection_memory_min_tokens",
    "qwen_exo_dream_reflection_max_attempts": "qwen_exo_reflection_memory_max_attempts",
    "qwen_exo_dream_reflection_max_output_tokens": "qwen_exo_reflection_memory_max_output_tokens",
    "qwen_exo_dream_reflection_max_history_tokens": "qwen_exo_reflection_memory_max_history_tokens",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _revision(values: dict[str, Any]) -> str:
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _coerce(setting: ServiceSetting, value: Any) -> bool | int | float | str:
    if setting.value_type == "boolean":
        if type(value) is not bool:
            raise ServiceConfigError("invalid_type", "必须是布尔值", field=setting.key)
        coerced: bool | int | float | str = value
    elif setting.value_type == "integer":
        if type(value) is not int:
            raise ServiceConfigError("invalid_type", "必须是整数", field=setting.key)
        coerced = value
    elif setting.value_type == "number":
        if type(value) not in {int, float}:
            raise ServiceConfigError("invalid_type", "必须是数字", field=setting.key)
        coerced = float(value)
    else:
        if type(value) is not str:
            raise ServiceConfigError("invalid_type", "必须是字符串", field=setting.key)
        coerced = value

    if setting.choices and coerced not in setting.choices:
        raise ServiceConfigError(
            "invalid_choice",
            f"必须是以下值之一：{', '.join(setting.choices)}",
            field=setting.key,
        )
    if setting.minimum is not None and coerced < setting.minimum:
        raise ServiceConfigError(
            "below_minimum", f"不得小于 {setting.minimum}", field=setting.key
        )
    if setting.maximum is not None and coerced > setting.maximum:
        raise ServiceConfigError(
            "above_maximum", f"不得大于 {setting.maximum}", field=setting.key
        )
    return coerced


def _validate_runtime_contract(values: dict[str, Any]) -> None:
    if values["max_prefill_tokens"] > values["context_length"]:
        raise ServiceConfigError(
            "invalid_relation",
            "Prefill 批量上限不能大于上下文长度",
            field="max_prefill_tokens",
        )
    max_compilable_document_tokens = values["context_length"] - 2048
    if max_compilable_document_tokens < 64:
        raise ServiceConfigError(
            "invalid_relation",
            "上下文长度必须至少容纳 2048 token 预留和一个 64-token Radix 页",
            field="context_length",
        )
    if (
        values["qwen_exo_tensor_bank_max_document_tokens"]
        > max_compilable_document_tokens
    ):
        raise ServiceConfigError(
            "invalid_relation",
            "文档长度上限不得大于上下文长度减 2048 token",
            field="qwen_exo_tensor_bank_max_document_tokens",
        )
    if values["qwen_exo_tensor_bank_salient_token_budget"] % 64:
        raise ServiceConfigError(
            "invalid_alignment",
            "显著片段预算必须按 64 token 对齐",
            field="qwen_exo_tensor_bank_salient_token_budget",
        )

    flags = QwenExoFeatureFlags(
        hybrid_prefix=values["qwen_exo_enable_hybrid_prefix"],
        external_memory=values["qwen_exo_enable_external_memory"],
        reference_judge=values["qwen_exo_enable_reference_judge"],
        capsule=values["qwen_exo_enable_capsule"],
        observer=values["qwen_exo_observer_mode"] != "off",
        adaptive_refresh=values["qwen_exo_enable_adaptive_refresh"],
        policy_data=values["qwen_exo_enable_policy_data"],
        score_bias=values["qwen_exo_score_bias_mode"] != "off",
    )
    kwargs = {
        setting.key.removeprefix("qwen_exo_"): values[setting.key]
        for setting in SERVICE_SETTINGS
        if setting.key.startswith("qwen_exo_")
    }
    for key in (
        "enable_hybrid_prefix",
        "enable_external_memory",
        "enable_policy_data",
        "enable_reference_judge",
        "enable_capsule",
        "enable_adaptive_refresh",
        "console_trace_default_scope",
    ):
        kwargs.pop(key, None)
    QwenExoConfig(
        state_directory=Path("/service-config-validation/state"),
        knowledge_directory=Path("/service-config-validation/knowledge"),
        model_path="service-config-validation",
        tp_size=2,
        max_running_requests=values["max_running_requests"],
        context_length=values["context_length"],
        feature_flags=flags,
        **kwargs,
    )


def validate_values(values: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(values) - set(_SETTINGS_BY_KEY))
    if unknown:
        raise ServiceConfigError(
            "unknown_setting", f"未知配置项：{', '.join(unknown)}", field=unknown[0]
        )
    missing = sorted(set(_SETTINGS_BY_KEY) - set(values))
    if missing:
        raise ServiceConfigError(
            "missing_setting", f"缺少配置项：{', '.join(missing)}", field=missing[0]
        )
    normalized = {
        key: _coerce(_SETTINGS_BY_KEY[key], value) for key, value in values.items()
    }
    try:
        _validate_runtime_contract(normalized)
    except ServiceConfigError:
        raise
    except ValueError as exc:
        raise ServiceConfigError("invalid_contract", str(exc)) from exc
    return normalized


def default_values() -> dict[str, Any]:
    return {setting.key: setting.default for setting in SERVICE_SETTINGS}


def _migrate_persisted_values(raw_values: object) -> dict[str, Any]:
    source = raw_values if isinstance(raw_values, dict) else {}
    known = {key: value for key, value in source.items() if key in _SETTINGS_BY_KEY}
    has_legacy_reflection_memory = any(
        key in source for key in _LEGACY_REFLECTION_MEMORY_KEYS
    )
    for legacy_key, current_key in _LEGACY_REFLECTION_MEMORY_KEYS.items():
        if current_key not in known and legacy_key in source:
            known[current_key] = source[legacy_key]
    if has_legacy_reflection_memory:
        for key in (
            "qwen_exo_qk_prefilter_mode",
            "qwen_exo_context_evidence_mode",
            "qwen_exo_context_integrity_mode",
            "qwen_exo_reflection_memory_mode",
            "qwen_exo_response_compaction_mode",
        ):
            known[key] = "active"
    for key in _SHADOW_TO_ACTIVE_KEYS:
        if known.get(key) == "shadow":
            known[key] = "active"
    return validate_values({**default_values(), **known})


def _value_from_args(setting: ServiceSetting, args: list[str]) -> Any | None:
    found: Any | None = None
    for index, token in enumerate(args):
        if setting.value_type == "boolean":
            if token == setting.flag:
                found = True
            elif token == setting.negative_flag:
                found = False
        elif token == setting.flag and index + 1 < len(args):
            raw = args[index + 1]
            if setting.value_type == "integer":
                found = int(raw)
            elif setting.value_type == "number":
                found = float(raw)
            else:
                found = raw
        elif token.startswith(setting.flag + "="):
            raw = token.split("=", 1)[1]
            if setting.value_type == "integer":
                found = int(raw)
            elif setting.value_type == "number":
                found = float(raw)
            else:
                found = raw
    return found


def values_from_args(args: Iterable[str]) -> dict[str, Any]:
    argv = list(args)
    values = default_values()
    for setting in SERVICE_SETTINGS:
        explicit = _value_from_args(setting, argv)
        if explicit is not None:
            values[setting.key] = explicit
    return validate_values(values)


def apply_values_to_args(args: Iterable[str], values: dict[str, Any]) -> list[str]:
    values = validate_values(values)
    argv = list(args)
    managed_flags = {
        flag
        for setting in SERVICE_SETTINGS
        for flag in (setting.flag, setting.negative_flag)
    }
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        plain_flag = token.split("=", 1)[0]
        if plain_flag in managed_flags:
            setting = next(
                item
                for item in SERVICE_SETTINGS
                if plain_flag in {item.flag, item.negative_flag}
            )
            index += 1
            if (
                setting.value_type != "boolean"
                and "=" not in token
                and index < len(argv)
            ):
                index += 1
            continue
        cleaned.append(token)
        index += 1

    for setting in SERVICE_SETTINGS:
        value = values[setting.key]
        if setting.value_type == "boolean":
            cleaned.append(setting.flag if value else setting.negative_flag)
        else:
            cleaned.extend((setting.flag, str(value)))
    return cleaned


class ServiceConfigStore:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def from_environment(cls) -> ServiceConfigStore:
        return cls(
            Path(os.getenv("QWEN_EXO_SERVICE_CONFIG", str(_DEFAULT_CONFIG_PATH)))
        )

    @property
    def managed_restart(self) -> bool:
        return os.getenv("QWEN_EXO_MANAGED_RESTART", "0") == "1"

    def _read_document(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ServiceConfigError(
                "config_unreadable", f"无法读取服务配置：{exc}"
            ) from exc
        if payload.get("schema") != _SERVICE_CONFIG_SCHEMA:
            raise ServiceConfigError("schema_mismatch", "服务配置 schema 不受支持")
        return payload

    def _write_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    document, stream, ensure_ascii=False, sort_keys=True, indent=2
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def ensure(self, base_args: Iterable[str]) -> dict[str, Any]:
        document = self._read_document()
        if document is None:
            values = values_from_args(base_args)
            revision = _revision(values)
            document = {
                "schema": _SERVICE_CONFIG_SCHEMA,
                "revision": revision,
                "applied_revision": None,
                "updated_at": _utc_now(),
                "applied_at": None,
                "healthy_revision": None,
                "healthy_at": None,
                "boot_attempts": 0,
                "previous_revision": None,
                "previous_values": None,
                "last_failed_revision": None,
                "last_rollback_at": None,
                "values": values,
            }
            self._write_document(document)
        else:
            values = _migrate_persisted_values(document.get("values"))
            if values != document.get("values"):
                document.update(
                    previous_revision=document.get("revision"),
                    previous_values=document.get("values"),
                    revision=_revision(values),
                    updated_at=_utc_now(),
                    values=values,
                    boot_attempts=0,
                )
                self._write_document(document)
        return document

    def mark_applied(
        self, base_args: Iterable[str]
    ) -> tuple[dict[str, Any], list[str]]:
        document = self.ensure(base_args)
        revision = document["revision"]
        if (
            document.get("healthy_revision") != revision
            and int(document.get("boot_attempts", 0)) >= 1
            and document.get("previous_revision")
            and document.get("previous_values")
        ):
            document["last_failed_revision"] = revision
            document["last_rollback_at"] = _utc_now()
            previous_values = _migrate_persisted_values(document.pop("previous_values"))
            document.pop("previous_revision")
            document["values"] = previous_values
            document["revision"] = _revision(previous_values)
            revision = document["revision"]

        if document.get("healthy_revision") == revision:
            document["boot_attempts"] = 0
        else:
            document["boot_attempts"] = int(document.get("boot_attempts", 0)) + 1
        document["applied_revision"] = revision
        document["applied_at"] = _utc_now()
        self._write_document(document)
        return document, apply_values_to_args(base_args, document["values"])

    def mark_healthy(self) -> bool:
        document = self._read_document()
        if document is None or document.get("applied_revision") != document.get(
            "revision"
        ):
            return False
        document["healthy_revision"] = document["revision"]
        document["healthy_at"] = _utc_now()
        document["boot_attempts"] = 0
        document["previous_revision"] = None
        document["previous_values"] = None
        self._write_document(document)
        return True

    def public_document(self) -> dict[str, Any]:
        document = self._read_document()
        if document is None:
            raise ServiceConfigError(
                "config_not_initialized", "服务配置尚未由托管启动器初始化"
            )
        values = _migrate_persisted_values(document.get("values"))
        return {
            "schema": _SERVICE_CONFIG_SCHEMA,
            "revision": document["revision"],
            "applied_revision": document.get("applied_revision"),
            "pending_restart": document.get("applied_revision") != document["revision"],
            "updated_at": document.get("updated_at"),
            "applied_at": document.get("applied_at"),
            "healthy_revision": document.get("healthy_revision"),
            "healthy_at": document.get("healthy_at"),
            "boot_attempts": int(document.get("boot_attempts", 0)),
            "last_failed_revision": document.get("last_failed_revision"),
            "last_rollback_at": document.get("last_rollback_at"),
            "managed_restart": self.managed_restart,
            "groups": list(_GROUPS),
            "settings": [setting.public_dict() for setting in SERVICE_SETTINGS],
            "values": values,
        }

    def update(
        self, updates: dict[str, Any], *, expected_revision: str | None
    ) -> dict[str, Any]:
        document = self._read_document()
        if document is None:
            raise ServiceConfigError(
                "config_not_initialized", "服务配置尚未由托管启动器初始化"
            )
        if expected_revision is not None and expected_revision != document["revision"]:
            raise ServiceConfigError(
                "revision_conflict", "配置已被其他会话更新，请刷新后重试"
            )
        unknown = sorted(set(updates) - set(_SETTINGS_BY_KEY))
        if unknown:
            raise ServiceConfigError(
                "unknown_setting", f"未知配置项：{', '.join(unknown)}", field=unknown[0]
            )
        values = _migrate_persisted_values(document.get("values"))
        values.update(updates)
        values = validate_values(values)
        revision = _revision(values)
        if revision != document["revision"]:
            document.update(
                previous_revision=document["revision"],
                previous_values=document["values"],
                revision=revision,
                updated_at=_utc_now(),
                values=values,
                boot_attempts=0,
            )
            self._write_document(document)
        return self.public_document()


def request_managed_restart(delay_seconds: float = 1.25) -> None:
    if os.getenv("QWEN_EXO_MANAGED_RESTART", "0") != "1":
        raise ServiceConfigError(
            "restart_unmanaged", "当前服务未由自动重启策略托管，拒绝写入不可生效的配置"
        )

    def terminate() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(delay_seconds, terminate)
    timer.daemon = True
    timer.start()
