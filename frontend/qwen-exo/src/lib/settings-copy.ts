import { translate } from "@/lib/i18n";

export type PlainSettingCopy = {
  label: string;
  summary: string;
};

export const SETTING_COPY: Record<string, readonly [string, string]> = {
  context_length: [
    "上下文长度",
    "定义单次请求可进入模型的最大 token 容量；调大可承载更长历史，调小可降低显存压力，当前双卡默认 131072（128K）。",
  ],
  default_enable_thinking: [
    "默认启用思考",
    "仅在客户端未显式指定推理选项时生效；关闭可避免普通请求自动进入 THINK，客户端显式开启仍然优先。",
  ],
  default_preserve_thinking: [
    "保留历史思考",
    "控制聊天模板是否保留历史 assistant 消息中的 THINK；关闭可减少上下文占用，回答正文不受影响。",
  ],
  mem_fraction_static: [
    "静态显存预留",
    "定义启动时为模型权重与 KV/GDN 缓存预留的显存比例；提高可增强容量，降低可提升余量，推荐 0.80。",
  ],
  max_running_requests: [
    "运行请求并发",
    "定义调度器同时执行的主请求上限；增大偏吞吐，单人低延迟建议 1～8，默认 64。",
  ],
  max_prefill_tokens: [
    "预填充批量上限",
    "定义单个调度批次累计读取的提示 token 上限；影响首字延迟与吞吐的平衡，推荐 65536。",
  ],
  qwen_exo_max_internal_fanout: [
    "内部任务并发",
    "限制反思记忆、上下文压缩、推理截断自问等后台任务的并行度，防止内部负载挤占主请求调度资源与 Mamba 状态槽；推荐 8。",
  ],
  qwen_exo_max_internal_tokens: [
    "内部任务 Token 预算",
    "限制每个请求的内部任务生成预算；过小会截断审计，过大会增加主请求延迟。开启 Responses 上下文压缩时必须覆盖摘要预算，推荐 2048。",
  ],
  qwen_exo_max_reasoning_tokens: [
    "默认 / 内部思考预算",
    "未指定 effort、数值 effort 和内部任务的思考 token 预算；不限制显式档位的独立预算，默认 3072。",
  ],
  qwen_exo_reasoning_budget_min: [
    "min 思考预算",
    "显式 min（或 minimal）档位的最大思考 token 数，仍受请求总输出与上下文限制；默认 512。",
  ],
  qwen_exo_reasoning_budget_low: [
    "low 思考预算",
    "显式 low 档位的最大思考 token 数，仍受请求总输出与上下文限制；默认 1024。",
  ],
  qwen_exo_reasoning_budget_medium: [
    "medium 思考预算",
    "显式 medium 档位的最大思考 token 数，仍受请求总输出与上下文限制；默认 2048。",
  ],
  qwen_exo_reasoning_budget_high: [
    "high 思考预算",
    "显式 high 档位使用此预算，即使模型模板映射到 xhigh；默认 4090。",
  ],
  qwen_exo_reasoning_budget_xhigh: [
    "xhigh 思考预算",
    "显式 xhigh（或 max）档位不受默认思考预算限制，仍受请求总输出与上下文限制；默认 8192。",
  ],
  qwen_exo_reasoning_mode: [
    "思考预算模式",
    "仅在关闭强制自动后生效：固定模式使用默认预算；动态模式为未指定 effort 的请求选档，并遵循任务链预算复用设置。此时显式 effort（包括 none）优先。",
  ],
  qwen_exo_force_auto_reasoning: [
    "强制自动思考预算",
    "默认开启：所有 Responses 请求使用自动档位，覆盖客户端档位（包括 none）和固定模式；是否重新调用 JEV 由任务链预算复用设置决定。关闭后恢复显式档位优先。",
  ],
  qwen_exo_reasoning_session_once: [
    "任务链思考预算复用",
    "默认开启：需要自动选档时，仅根据首次用户任务调用一次 JEV，后续工具轮复用该档位；新的真实用户任务或新会话重新评估。关闭后每个自动选档请求均调用 JEV；不改变显式 effort 的优先级。",
  ],
  qwen_exo_reasoning_dynamic_context_tokens: [
    "动态思考评估上下文",
    "内部动态档位评估最多读取的上下文 token 数，不改变主请求上下文；默认 2048。",
  ],
  qwen_exo_enable_hybrid_prefix: [
    "混合前缀状态",
    "同时管理 Full-Attention K/V 与 GDN 线性状态；QWEN-EXO 原生记忆必须开启，纯模型基线可关闭。",
  ],
  qwen_exo_enable_policy_data: [
    "策略数据",
    "允许策略文档以原生状态条件化工程行为；需要固定执行边界时开启，排查策略干扰时关闭。",
  ],
  qwen_exo_enable_capsule: [
    "执行胶囊",
    "跨轮持久化粗粒度执行状态；长任务建议开启，单轮无状态问答可关闭。",
  ],
  qwen_exo_max_policy_tokens: [
    "策略注入预算",
    "限制策略文档对应的原生状态 token 预算；普通策略推荐 4096，不建议为容纳无关规则扩大。",
  ],
  qwen_exo_observer_mode: [
    "观测模式",
    "定义解码期惊奇度与 Attention-Q 漂移的观测级别，观测信号供注意力分数偏置使用；主动干预允许触发自我提问，仅观测只记录，关闭最省开销。",
  ],
  qwen_exo_immediate_uncertainty_retrieval: [
    "即时自我提问",
    "观测命中后立即启动隐藏的自我提问；纠错速度优先时开启，内部请求过多时可关闭。",
  ],
  qwen_exo_observer_surprisal_threshold: [
    "局部惊奇度阈值",
    "定义最近 token 局部平均惊奇度的触发下限；降低更敏感，提高更安静，推荐 0.8。",
  ],
  qwen_exo_observer_surprisal_window: [
    "惊奇度窗口",
    "定义局部惊奇度统计使用的最近 token 数；窗口小响应快但噪声高，窗口大更稳定，推荐 8。",
  ],
  qwen_exo_observer_surprisal_margin: [
    "惊奇度增量",
    "要求局部惊奇度相对历史基线超过指定增量才触发，减少缓慢波动误报，推荐 0.2。",
  ],
  qwen_exo_observer_q_drift_threshold: [
    "查询漂移阈值",
    "定义当前 Attention-Q 相对近期方向的差异触发下限；降低更敏感，提高减少误触发，推荐 0.35。",
  ],
  qwen_exo_observer_cooldown_tokens: [
    "观测冷却间隔",
    "定义同一请求两次观测干预的最小 token 间隔；重复触发较多时提高，推荐 64。",
  ],
  qwen_exo_observer_max_triggers: [
    "单次触发上限",
    "限制一次请求允许的自动干预次数；当前安全上限为 1，0 表示仅观测不触发。",
  ],
  qwen_exo_observer_q_pre_tokens: [
    "触发前查询快照",
    "定义触发点前保留的 Attention-Q 快照数量，用于恢复偏航前的关注上下文，推荐 8。",
  ],
  qwen_exo_observer_q_post_tokens: [
    "触发后查询快照",
    "定义触发点后继续采集的 Attention-Q 快照数量，用于确认不确定性是否持续，推荐 4。",
  ],
  qwen_exo_observer_recovery_tokens: [
    "恢复判定窗口",
    "定义触发后观察多少个未来 token 再判定不确定性恢复；较大更谨慎，较小更快进入纠错，推荐 8。",
  ],
  qwen_exo_score_bias_mode: [
    "注意力控制模式",
    "定义历史轨迹是否参与注意力加权；仅观察只记录选择，主动控制施加上限内偏置，关闭用于纯基线。",
  ],
  qwen_exo_score_bias_min_surprisal: [
    "历史片段信息量下限",
    "过滤低惊奇度、模板化历史片段；值越高候选越少但质量更高，推荐 0.8。",
  ],
  qwen_exo_score_bias_max: [
    "注意力偏置上限",
    "定义历史轨迹可获得的绝对注意力加权上限，防止历史覆盖当前输入，推荐 0.25。",
  ],
  qwen_exo_score_bias_half_life_steps: [
    "历史半衰期",
    "定义轨迹权重随会话轮次指数衰减的半衰期；增大延长历史影响，减小更快遗忘，推荐 4 轮。",
  ],
  qwen_exo_score_bias_max_blocks: [
    "历史候选块上限",
    "限制进入注意力控制计算的历史块总数，以控制显存、时延和选择噪声，推荐 8。",
  ],
  qwen_exo_score_bias_min_age_steps: [
    "历史最小轮次年龄",
    "要求轨迹至少经过指定轮次后才参与注意力控制，降低短期回声与重复，推荐 2。",
  ],
  qwen_exo_score_bias_max_age_steps: [
    "历史最大轮次年龄",
    "超过该轮次的历史块退出候选，避免过期任务状态重新吸引注意力，推荐 16。",
  ],
  qwen_exo_score_bias_tail_tokens: [
    "轨迹尾部扫描上限",
    "限制仅从最近若干 token 的轨迹尾部构造候选块，控制扫描成本，推荐 4096。",
  ],
  qwen_exo_score_bias_tail_ratio: [
    "轨迹尾部扫描比例",
    "按轨迹总长进一步截取尾部比例；较小偏向近期步骤，较大覆盖更长过程，推荐 15%。",
  ],
  qwen_exo_score_bias_selected_blocks: [
    "实际增强块数",
    "定义候选排序后真正施加注意力控制的最大历史块数；增加扩大覆盖，但可能引入噪声，推荐 2。",
  ],
  qwen_exo_score_bias_query_window: [
    "轨迹相关性查询窗口",
    "定义汇总当前 Attention-Q 的最近 token 窗口；窗口小更灵敏，窗口大更稳定，推荐 8。",
  ],
  qwen_exo_score_bias_min_relevance: [
    "轨迹相关性下限",
    "要求历史块与当前查询的余弦相关性达到该值才参与控制；默认 0.01，设为 0 可关闭正相关门槛。",
  ],
  qwen_exo_score_bias_relevance_margin: [
    "轨迹相关性边际",
    "要求最佳历史块至少领先后续候选该差值，避免在含糊候选之间强行选择，推荐 0.005。",
  ],
  qwen_exo_score_bias_anchor_bias: [
    "系统/工具锚点偏置",
    "对 system instructions 与工具 schema 的少量 token span 提供有界保护；默认 0.01，设为 0 才关闭。",
  ],
  qwen_exo_score_bias_anchor_max_blocks: [
    "系统/工具锚点块数",
    "限制 system instructions 与工具 schema 参与 decode 锚定的最多 128-token 块数。",
  ],
  qwen_exo_reflection_memory_mode: [
    "反思记忆",
    "工具轨迹空闲后提炼可复用经验，作为文本文件写入知识库 reflection-memory/ 目录，供记忆工具检索；生成失败或证据不足时不写入。",
  ],
  qwen_exo_response_compaction_mode: [
    "Responses 上下文压缩",
    "启用显式 /v1/responses/compact 请求的有界摘要与前一轮混合状态复用。",
  ],
  qwen_exo_telemetry_text_mode: [
    "遥测原文记录",
    "定义遥测事件保存请求与输出原文的范围；仅编辑片段兼顾审计与体积，全量记录仅用于短期排障。",
  ],

  qwen_exo_console_trace_default_scope: [
    "召回轨迹默认范围",
    "定义召回轨迹控制台的默认显示范围：记忆活动保留调用过记忆工具或有内部活动的请求，仅实际召回只保留调用过记忆工具的请求，全部请求用于完整审计。",
  ],

  qwen_exo_activation_editor_strength: [
    "轨迹编辑器强度",
    "由用户按任务选择标准、明显或最强档位；系统不会根据训练结果自动改档。",
  ],
};

export function settingCopy(
  key: string,
  fallbackLabel: string,
  fallbackSummary: string,
): PlainSettingCopy {
  const value = SETTING_COPY[key];
  return {
    label: translate(value?.[0] || fallbackLabel),
    summary: translate(value?.[1] || fallbackSummary),
  };
}
