from __future__ import annotations

import json


def render_recall_trace_html(payload):
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    template = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>QWEN EXO / 飞行中召回轨迹</title>
<style>
:root{color-scheme:dark;--ink:#080c0d;--panel:#101617;--line:#293334;--paper:#e9efea;--muted:#8a9998;--dim:#596665;--lime:#d6f45f;--cyan:#55ded3;--amber:#ffbd5c;--red:#ff6c61;font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","Segoe UI",sans-serif;background:var(--ink);color:var(--paper)}*{box-sizing:border-box}body{margin:0;min-width:320px;background:radial-gradient(circle at 12% -20%,rgba(85,222,211,.11),transparent 38rem),var(--ink)}button,select,input{color:inherit;font:inherit}button{cursor:pointer}a{color:inherit}main{width:min(1420px,100%);margin:auto;padding:28px}.mast{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding:18px 0 24px}.mast p,.eyebrow{margin:0 0 8px;color:var(--cyan);font:700 10px/1 ui-monospace,monospace;letter-spacing:.15em}.mast h1{margin:0;font-size:clamp(28px,4vw,48px);line-height:1;letter-spacing:-.05em}.mast nav{display:flex;gap:8px}.mast a,.toolbar button{border:1px solid var(--line);padding:9px 12px;color:var(--muted);font-size:11px;text-decoration:none;background:transparent}.mast a:hover{border-color:var(--lime);color:var(--lime)}.toolbar,.card{border:1px solid var(--line);background:rgba(16,22,23,.94)}.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0;padding:13px 15px}.toolbar label{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:11px}.toolbar select{min-width:280px;border:1px solid #465354;border-radius:0;padding:8px;background:var(--ink)}.badge{display:inline-block;border:1px solid var(--line);padding:5px 7px;color:var(--muted);font:600 9px/1 ui-monospace,monospace}.toolbar .danger{margin-left:auto;border-color:#743a36;color:#ffaaa3}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.card{min-width:0;padding:16px}.card h2{margin:0 0 14px;font-size:15px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line)}.metric{min-height:82px;padding:12px;background:var(--panel)}.metric span{display:block;color:var(--dim);font-size:9px}.metric b{display:block;margin-top:12px;overflow:hidden;color:var(--paper);font:650 14px/1.2 ui-monospace,monospace;text-overflow:ellipsis;white-space:nowrap}.candidate{display:grid;grid-template-columns:30px minmax(0,1fr) 70px;align-items:center;gap:10px;border-top:1px solid var(--line);padding:10px 0}.candidate:first-child{border-top:0}.candidate>span{min-width:0}.candidate b{color:var(--cyan);font:650 10px/1 ui-monospace,monospace}.candidate .name{overflow:hidden;font-size:11px;text-overflow:ellipsis;white-space:nowrap}.candidate small{display:block;margin-top:5px;color:var(--dim);font-size:9px}.decision{font:700 9px/1 ui-monospace,monospace;text-align:right}.decision.yes{color:var(--lime)}.decision.no{color:var(--red)}.policy-list{display:flex;gap:7px;flex-wrap:wrap}.policy-list span{border:1px solid #5b4a29;padding:7px 9px;color:var(--amber);font-size:10px}.event{margin-top:10px;border-left:2px solid var(--cyan);padding:13px 14px;background:#0c1213}.event.is-admitted{border-left-color:var(--lime)}.event.is-rejected{border-left-color:var(--red)}.event h3{margin:0 0 10px;font-size:12px}.event p{margin:8px 0;color:var(--muted);font-size:10px;line-height:1.65}.event .badge{margin:0 5px 5px 0}.event .question{border-top:1px solid var(--line);padding-top:9px;color:var(--paper)}.empty{border:1px dashed var(--line);padding:20px;color:var(--dim);font-size:11px;text-align:center}.raw{grid-column:1/-1}.raw details summary{color:var(--muted);font-size:11px;cursor:pointer}.raw pre{max-height:520px;overflow:auto;margin:13px 0 0;padding:14px;color:#aab8b4;background:#090d0e;font:10px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.split{display:grid;grid-template-columns:1fr 1fr;gap:10px}.fact{border-top:1px solid var(--line);padding:9px 0}.fact span{display:block;color:var(--dim);font-size:9px}.fact b{display:block;margin-top:5px;font:600 10px/1.4 ui-monospace,monospace}.toast{position:fixed;right:20px;bottom:20px;border:1px solid var(--lime);padding:11px 14px;color:var(--ink);background:var(--lime);font-size:11px}@media(max-width:900px){main{padding:16px}.grid{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,1fr)}.raw{grid-column:auto}.mast{align-items:flex-start;flex-direction:column}.toolbar select{min-width:0;max-width:100%}}@media(max-width:520px){.split{grid-template-columns:1fr}.candidate{grid-template-columns:25px minmax(0,1fr)}.candidate .decision{grid-column:2;text-align:left}.toolbar .danger{margin-left:0}.metric-grid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<main>
<header class="mast"><div><p>QWEN EXO / 运行诊断</p><h1>飞行中召回轨迹</h1></div><nav><a href="/qwen-exo/">用户工作台</a><a href="/qwen-exo/admin">管理控制台</a></nav></header>
<div id="app"></div>
</main>
<script>
const data=__DATA__,app=document.getElementById("app");
const esc=value=>String(value??"").replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const fmt=(value,digits=3)=>value===null||value===undefined?"—":Number(value).toFixed(digits);
const labels={not_scheduled:"未调度",not_compiled:"未编译",ready_for_safe_replay:"可安全重放",replay_admitted:"重放已准入",replay_rejected:"重放已拒绝",reject_fail_closed:"失败关闭拒绝",admit_maybe:"Maybe 已准入",observer_shadow:"影子观测",pending:"等待判定"};
const label=value=>labels[String(value||"")]||String(value||"未知");
function candidateRows(items){
  if(!(items||[]).length)return '<div class="empty">本轮没有提出知识候选</div>';
  return items.map((item,index)=>{const support=item.semantic_support||{},status=support.supported===true?"yes":support.supported===false?"no":"",decision=support.supported===true?"已准入":support.supported===false?"已拒绝":"待判定";return `<article class="candidate"><b>#${index+1}</b><span><span class="name">${esc(item.document||item.document_id||item.candidate_id)}</span><small>相关度 ${fmt(item.score,4)} · ${item.policy?"策略候选":"知识候选"}</small></span><strong class="decision ${status}">${decision}</strong></article>`}).join("");
}
function policyRows(turn){
  const ids=turn.policy_data_document_ids||[],documents=data.bank?.policy_documents||[];
  if(!ids.length)return '<div class="empty">本轮没有通过判定的 PolicyData 策略</div>';
  return `<div class="policy-list">${ids.map(id=>{const info=documents.find(item=>item.document_id===id);return `<span>${esc(info?.relative_path||id)}</span>`}).join("")}</div>`;
}
function eventRows(events){
  if(!events.length)return '<div class="empty">本轮没有解码观测召回事件</div>';
  return events.map((item,index)=>{const admitted=item.maybe_decision==="admit_maybe",rejected=String(item.replay_decision||"").startsWith("reject"),questions=item.self_ask_questions||[];return `<article class="event ${admitted?"is-admitted":rejected?"is-rejected":""}"><h3>召回 ${index+1} · 令牌 ${item.token_index??"—"} · 层 ${item.layer??"—"}</h3><p><span class="badge">惊奇度 ${fmt(item.latest_surprisal)}</span><span class="badge">窗口均值 ${fmt(item.window_mean)}</span><span class="badge">历史均值 ${fmt(item.history_mean)}</span><span class="badge">${esc(label(item.replay_decision))}</span><span class="badge">${esc(label(item.maybe_decision))}</span></p>${questions.length?`<p class="question"><b>模型自问：</b>${questions.map(esc).join("；")}</p>`:""}</article>`}).join("");
}
function metrics(turn){
  const timing=turn.timing||{},stage=turn.think_recall_stage_timing||{},events=turn.think_recall_events||[],candidates=turn.knowledge_candidates||[],admitted=(turn.selected_document_ids||[]).length+(turn.policy_data_document_ids||[]).length;
  const values=[["推理策略",turn.strategy||"—"],["总耗时",`${fmt(timing.total_seconds,2)} 秒`],["知识候选",candidates.length],["通过判定",admitted],["策略令牌",turn.policy_data_tokens||0],["自问耗时",`${fmt(stage.self_ask_seconds,2)} 秒`],["重放耗时",`${fmt(stage.exact_replay_seconds,2)} 秒`],["解码事件",events.length]];
  return values.map(([name,value])=>`<div class="metric"><span>${name}</span><b title="${esc(value)}">${esc(value)}</b></div>`).join("");
}
function render(index){
  if(!data.turns.length){app.innerHTML='<div class="toolbar"><b>0 个轮次</b></div><div class="card empty">暂无轨迹。新的请求完成后会自动记录。</div>';return}
  const turn=data.turns[index]||data.turns.at(-1),events=turn.think_recall_events||[],candidates=turn.knowledge_candidates||[],refresh=turn.pending_maybe_compilation||{};
  app.innerHTML=`<div class="toolbar"><label>轮次<select id="turn">${data.turns.map((item,itemIndex)=>`<option value="${itemIndex}" ${itemIndex===index?"selected":""}>${itemIndex+1} · ${esc(item.trajectory_id||item.turn_id||"")}</option>`).join("")}</select></label><span class="badge">${data.turns.length} 个轮次</span><span class="badge">架构 ${esc(data.schema)}</span><button id="clear" class="danger" type="button">清空全部轨迹</button></div><section class="card"><p class="eyebrow">本轮摘要</p><div class="metric-grid">${metrics(turn)}</div></section><div class="grid"><section class="card"><h2>PolicyData 策略准入</h2>${policyRows(turn)}</section><section class="card"><h2>Knowledge 知识判定</h2>${candidateRows(candidates)}</section><section class="card"><h2>解码阶段召回</h2>${eventRows(events)}</section><section class="card"><h2>恢复与重放状态</h2><div class="split"><div class="fact"><span>父响应</span><b>${esc(turn.parent_response_id||"无")}</b></div><div class="fact"><span>记忆条件</span><b>${esc(turn.memory_condition||"—")}</b></div><div class="fact"><span>刷新判定</span><b>${esc(label(refresh.status||"not_scheduled"))}</b></div><div class="fact"><span>外部学习</span><b>${turn.external_learning_restored?"已恢复":"未启用"}</b></div></div></section><section class="card raw"><details><summary>查看本轮原始诊断数据</summary><pre>${esc(JSON.stringify(turn,null,2))}</pre></details></section></div>`;
  document.getElementById("turn").onchange=event=>render(Number(event.target.value));
  document.getElementById("clear").onclick=clearTrace;
}
async function clearTrace(){
  if(!confirm(`确定清空全部 ${data.turns.length} 条轨迹吗？此操作不可撤销。`))return;
  const button=document.getElementById("clear");button.disabled=true;button.textContent="正在清空…";
  try{const response=await fetch("/qwen-exo/recall-trace",{method:"DELETE"});if(!response.ok){const payload=await response.json().catch(()=>({}));throw new Error(payload.detail||`HTTP ${response.status}`)}data.turns.length=0;render(0)}catch(error){button.disabled=false;button.textContent="清空全部轨迹";alert(`清空失败：${error.message}`)}
}
render(Math.max(data.turns.length-1,0));
</script>
</body>
</html>"""
    return template.replace("__DATA__", encoded)
