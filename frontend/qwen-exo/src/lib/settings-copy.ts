import { translate } from "@/lib/i18n";

export type PlainSettingCopy = {
  label: string;
  summary: string;
};

export const SETTING_COPY: Record<string, readonly [string, string]> = {
  context_length: [
    "上下文长度",
    "定义单次请求可进入模型的最大 token 容量；调大可承载更长历史，调小可降低显存压力，当前双卡推荐 102400。",
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
    "限制语义准入、Self-Ask 等内部任务的并行度，防止后台负载挤占主请求调度资源；推荐 32。",
  ],
  qwen_exo_max_internal_tokens: [
    "内部任务 Token 预算",
    "限制每个请求的内部任务生成预算；过小会截断审计，过大会增加主请求延迟。开启 Responses 上下文压缩时必须覆盖摘要预算，推荐 2048。",
  ],
  qwen_exo_max_reasoning_tokens: [
    "推理思考上限",
    "限制隐藏推理阶段的最大 token 数；复杂编码任务可提高，延迟敏感任务可降低，推荐 3072。",
  ],
  qwen_exo_enable_hybrid_prefix: [
    "混合前缀状态",
    "同时管理 Full-Attention K/V 与 GDN 线性状态；QWEN-EXO 原生记忆必须开启，纯模型基线可关闭。",
  ],
  qwen_exo_enable_external_memory: [
    "知识库记忆",
    "允许请求检索并恢复知识文档的原生状态；知识增强场景开启，无外部知识基线关闭。",
  ],
  qwen_exo_enable_policy_data: [
    "策略数据",
    "允许策略文档以原生状态条件化工程行为；需要固定执行边界时开启，排查策略干扰时关闭。",
  ],
  qwen_exo_enable_reference_judge: [
    "语义准入",
    "对 Q×K 提出的候选执行语义相关性复核；生产建议开启，仅做低延迟检索实验时可关闭。",
  ],
  qwen_exo_enable_capsule: [
    "执行胶囊",
    "跨轮持久化粗粒度执行状态；长任务建议开启，单轮无状态问答可关闭。",
  ],
  qwen_exo_max_candidates: [
    "候选数量上限",
    "定义每次检索进入后续准入的候选文档数；提高增强召回覆盖，降低减少语义审计耗时，推荐 8。",
  ],
  qwen_exo_qk_recall_preset: [
    "Q×K 召回严格度",
    "固定设置第一道 Q×K 门禁的分数与领先差距要求：高召回偏覆盖，标准保持平衡，高精度优先减少误召回。",
  ],
  qwen_exo_max_memory_tokens: [
    "知识注入预算",
    "限制单次请求允许准入的知识 token 总量；防止长文档挤占对话上下文，推荐 8192。",
  ],
  qwen_exo_max_policy_tokens: [
    "策略注入预算",
    "限制策略文档对应的原生状态 token 预算；普通策略推荐 4096，不建议为容纳无关规则扩大。",
  ],
  qwen_exo_qk_expansion_margin: [
    "检索扩展边距",
    "当前二名候选差距低于该值时扩大候选池重新排名；值越小越收敛，推荐 0.01。",
  ],
  qwen_exo_qk_only_knowledge: [
    "仅原生 Q×K 检索",
    "限制知识候选只能来自 Attention-Q × K 原生检索，关闭词法补充可提高一致性，但可能降低召回。",
  ],
  qwen_exo_tensor_bank_max_document_tokens: [
    "文档编译长度上限",
    "定义单份文档可编译入 Tensor Bank 的最大 token 数；普通文档推荐 4096，长轨迹实验才提高。",
  ],
  qwen_exo_tensor_bank_salient_token_budget: [
    "显著片段预算",
    "限制长文档保留为精确 K/V 的高信息 token 总量；提高可增强事实覆盖，降低可减少上下文占用。",
  ],
  qwen_exo_tensor_bank_surprisal_threshold: [
    "显著度阈值",
    "以模型惊奇度筛选高信息 token；提高只保留更罕见内容，降低会纳入更多普通片段，推荐 6.0。",
  ],
  qwen_exo_tensor_bank_span_tokens: [
    "显著片段半径",
    "定义每个高信息 token 保留的前后上下文宽度；半径越大语义越完整，但更容易超预算，推荐 16。",
  ],
  qwen_exo_observer_mode: [
    "观测模式",
    "定义解码期惊奇度与 Attention-Q 漂移的观测级别；主动干预可触发 Self-Ask，仅观测只记录，关闭最省开销。",
  ],
  qwen_exo_enable_adaptive_refresh: [
    "自适应刷新",
    "允许持续不确定时重新检索并语义审计候选；复杂代理任务建议开启，误触发较多时先调高观测门槛。",
  ],
  qwen_exo_immediate_uncertainty_retrieval: [
    "即时自我提问",
    "观测命中后立即启动隐藏 Self-Ask；纠错速度优先时开启，内部请求过多时可关闭。",
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
  qwen_exo_replay_observation_tokens: [
    "回放观测窗口",
    "定义基线分支与候选分支共同评分的未来 token 数；越多比较越稳定，推荐 8。",
  ],
  qwen_exo_replay_prefix_tokens: [
    "回放父前缀长度",
    "定义回放分支复用的原请求前缀 token 数；上下文依赖强时提高，资源受限时降低，推荐 1024。",
  ],
  qwen_exo_replay_max_candidates: [
    "回放候选数量",
    "定义一次因果回放比较的最大候选分支数；每增加一份都会增加内部评分开销，推荐 2。",
  ],
  qwen_exo_replay_reference_tokens: [
    "回放参考预算",
    "限制每条回放分支注入的参考 token 数，避免长参考改变输出分布，推荐 128。",
  ],
  qwen_exo_replay_minimum_gain: [
    "最小损失增益",
    "要求候选分支相对基线的损失改善达到该值才视为有效；调高更保守，推荐 0.02。",
  ],
  qwen_exo_replay_switch_margin: [
    "候选切换边际",
    "要求新候选在已有胜出者基础上额外领先该幅度，避免微小噪声造成反复切换，推荐 0.05。",
  ],
  qwen_exo_replay_maybe_kl_cap: [
    "分布差异上限",
    "限制候选分支相对基线的 KL 差异，防止强干预造成分布失真，推荐 4.0。",
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
    "要求历史块与当前查询的余弦相关性达到该值才参与控制，推荐 0.0。",
  ],
  qwen_exo_score_bias_relevance_margin: [
    "轨迹相关性边际",
    "要求最佳历史块至少领先后续候选该差值，避免在含糊候选之间强行选择，推荐 0.005。",
  ],
  qwen_exo_context_evidence_mode: [
    "工具后上下文证据",
    "外部知识均未通过准入时，只让最新工具结果中的明确事实作为当前请求的临时证据；不能证实时不注入。",
  ],
  qwen_exo_context_integrity_mode: [
    "完整性检查",
    "每次工具结果到达后审查私有上下文、工具账本与执行胶囊；只准入带当前工具原文证据的更正。",
  ],
  qwen_exo_reflection_memory_mode: [
    "反思记忆",
    "工具轨迹空闲后提炼可复用经验并热写入知识库；生成失败或证据不足时不写入。",
  ],
  qwen_exo_response_compaction_mode: [
    "Responses 上下文压缩",
    "启用显式 /v1/responses/compact 请求的有界摘要与前一轮混合状态复用。",
  ],
  qwen_exo_telemetry_text_mode: [
    "遥测原文记录",
    "定义遥测事件保存请求、输出与 Self-Ask 原文的范围；仅编辑片段兼顾审计与体积，全量记录仅用于短期排障。",
  ],

  qwen_exo_console_trace_default_scope: [
    "召回轨迹默认范围",
    "定义召回轨迹控制台的默认显示范围：记忆活动保留注入或 Self-Ask 请求，仅实际召回更严格，全部请求用于完整审计。",
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
