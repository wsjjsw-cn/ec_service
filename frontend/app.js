const API_BASE = window.location.origin.startsWith("file")
  ? "http://127.0.0.1:8000"
  : window.location.origin;

const prompts = [
  "你是谁",
  "七天无理由退货需要满足什么条件？",
  "AeroPods Pro 2 现在还有库存吗？",
  "退款一般多长时间到账",
  "如何维修商品",
  "讲个笑话",
  "谢谢"
];

// 后端未连接时的占位指标：一律展示 "--"，不臆造任何具体数值。
const FALLBACK_METRICS = { offline: true };

// 离线演示用回答：仅在无法连接后端时使用，便于静态预览完整交互。
const FALLBACK_ANSWERS = [
  {
    test: /^(你|您)是(谁|啥|什么|哪位|做什么)/,
    response: {
      answer: "您好，我是智能电商客服助手。可以帮您解答商品咨询、售后政策、订单查询、物流追踪等问题。请问有什么可以帮您？",
      route: "identity", cache_level: null, decision: null, sources: [], latency_ms: 1.2, llm_calls: 0,
      debug: { intent: "identity", mode: "offline-demo" }
    }
  },
  {
    test: /^(你好|您好|hello|hi)/i,
    response: {
      answer: "您好！我是智能电商客服助手，请问有什么可以帮您？",
      route: "chitchat", cache_level: null, decision: null, sources: [], latency_ms: 1.0, llm_calls: 0,
      debug: { intent: "chitchat", mode: "offline-demo" }
    }
  },
  {
    test: /^(谢谢|感谢|再见|拜拜|bye)/i,
    response: {
      answer: "不客气，很高兴能帮到您！如果还有其他问题，随时可以问我。",
      route: "chitchat", cache_level: null, decision: null, sources: [], latency_ms: 1.0, llm_calls: 0,
      debug: { intent: "chitchat", mode: "offline-demo" }
    }
  },
  {
    test: /讲个笑话|有什么游戏|电影推荐/,
    response: {
      answer: "抱歉，我主要是电商客服助手，只能解答商品、订单、售后、物流等购物相关问题。您的问题不在我的服务范围内，建议您咨询相关客服或查看帮助中心。",
      route: "ood", cache_level: null, decision: null, sources: [], latency_ms: 2.3, llm_calls: 0,
      debug: { intent: "out_of_domain", mode: "offline-demo" }
    }
  },
  {
    test: /库存|有货|AeroPods/i,
    response: {
      answer: "AeroPods Pro 2 蓝牙耳机 当前有货，库存为 12 件。库存数据更新时间：2026-08-01T10:00:00+08:00，下单前建议以商品页实时库存为准。",
      route: "dynamic", cache_level: null, decision: null, sources: [], latency_ms: 18.4, llm_calls: 0,
      debug: { guardrail: "stock", mode: "offline-demo" }
    }
  },
  {
    test: /退款|到账/,
    response: {
      answer: "退款在售后审核通过后原路退回。银行卡通常 1~5 个工作日到账，支付宝和微信通常 24 小时内到账，具体以支付机构处理时间为准。",
      route: "cache", cache_level: "L1", decision: "reuse", sources: ["faq_refund_time"], latency_ms: 12.8, llm_calls: 0,
      debug: { reason: "l1_intent_shortcut", score: 0.98, mode: "offline-demo" }
    }
  },
  {
    test: /退货|七天/,
    response: {
      answer: "七天无理由退货需在签收次日起 7 天内发起，商品、配件、赠品、包装和发票保持完好，不影响二次销售。定制、生鲜、拆封后影响安全或卫生的商品不支持无理由退货。",
      route: "cache", cache_level: "L1", decision: "reuse", sources: ["faq_return_7d"], latency_ms: 11.6, llm_calls: 0,
      debug: { reason: "l1_exact_normalized", score: 1.0, mode: "offline-demo" }
    }
  },
  {
    test: /如何维修|保修政策/i,
    response: {
      answer: "家电、数码等商品按国家三包和品牌政策提供保修。保修期内出现质量问题可申请免费维修；人为损坏、进水、私自拆修不在免费保修范围内。",
      route: "rag", cache_level: null, decision: "rag", sources: ["kb_warranty", "kb_repair"], latency_ms: 45.2, llm_calls: 1,
      debug: { reason: "full_rag_retrieval", mode: "offline-demo" }
    }
  }
];

const ROUTE_META = {
  dynamic: ["动态拦截", "实时性问题已绕过静态缓存"],
  identity: ["身份识别", "返回客服自我介绍"],
  chitchat: ["闲聊回应", "问候 / 感谢 / 告别等礼貌回复"],
  ood: ["领域外问题", "超出电商服务范围，已友好引导"],
  cache: ["缓存命中", "答案来自 L1 / L2 缓存或补全生成"],
  rag: ["完整 RAG", "缓存未命中，进入 ReAct 检索与生成"]
};

const state = { online: false, lastResponse: null };

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const messagesEl = $("#messages");
const inputEl = $("#questionInput");

const fmtPct = (v) => `${(Number(v) * 100).toFixed(2)}%`;
const clampPct = (base, now) => (base ? (base - now) / base : null);

function setStatus(mode) {
  state.online = mode === "online";
  const node = $("#serviceStatus");
  node.classList.remove("is-online", "is-offline", "is-pending");
  node.classList.add(mode === "online" ? "is-online" : mode === "offline" ? "is-offline" : "is-pending");
  node.querySelector(".status-text").textContent =
    mode === "online" ? "后端在线" : mode === "offline" ? "离线预览" : "检测服务中";
  const hint = $("#backendHint");
  if (hint) hint.textContent = mode === "online" ? "已连接后端" : "离线预览";
}

function addMessage(role, text, meta = null) {
  const item = document.createElement("div");
  item.className = `message ${role}`;

  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  if (meta) bubble.appendChild(meta);

  if (role === "user") item.append(bubble, avatar);
  else item.append(avatar, bubble);

  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function metaTags(parts) {
  const wrap = document.createElement("div");
  wrap.className = "message-meta";
  parts.filter(Boolean).forEach(({ label, cls }) => {
    const tag = document.createElement("span");
    tag.className = `tag ${cls || ""}`.trim();
    tag.textContent = label;
    wrap.appendChild(tag);
  });
  return wrap;
}

function renderPipeline(response) {
  $$("#pipeline li").forEach((li) => li.classList.remove("is-active"));
  if (!response) return;
  const route = response.route;
  let active = [];
  if (route === "dynamic") active = ["guardrail"];
  else if (route === "cache") {
    if (response.cache_level === "L1") active = ["l1"];
    else if (response.cache_level === "L2") active = ["l2", "validator"];
    else active = ["validator"];
  } else if (route === "rag") active = ["react"];
  active.forEach((stage) => {
    const el = document.querySelector(`#pipeline li[data-stage="${stage}"]`);
    if (el) el.classList.add("is-active");
  });
}

function renderDetail(response) {
  $("#routeBadge").textContent = response ? response.route || "-" : "等待请求";
  $("#detailGrid").innerHTML = `
    <div><dt>缓存层</dt><dd>${response?.cache_level || "-"}</dd></div>
    <div><dt>决策</dt><dd>${response?.decision || "-"}</dd></div>
    <div><dt>延迟</dt><dd>${Number(response?.latency_ms || 0).toFixed(2)} ms</dd></div>
    <div><dt>LLM 调用</dt><dd>${response?.llm_calls ?? 0}</dd></div>
  `;
  $("#traceBox").textContent = JSON.stringify(
    { route: response?.route, cache_level: response?.cache_level, sources: response?.sources, debug: response?.debug },
    null, 2
  );
}

function tile(label, value, sub, good = false) {
  return `<div class="metric ${good ? "is-good" : ""}"><span>${label}</span><strong>${value}</strong><small>${sub}</small></div>`;
}

function renderMetrics(report = FALLBACK_METRICS) {
  const grid = $("#metricsGrid");
  const note = $("#metricsNote");

  if (!report || report.offline) {
    grid.innerHTML = [
      tile("缓存覆盖率", "--", "冷启动"),
      tile("直接复用率", "--", "冷启动"),
      tile("LLM 调用节省", "--", "对照无缓存"),
      tile("Token 节省", "--", "对照无缓存"),
      tile("动态拦截", "--", "实时问题"),
      tile("输入 Token 节省", "--", "对照无缓存")
    ].join("");
    note.textContent = "未连接后端：指标需启动服务后由 /metrics/eval 实测生成（或运行 python scripts/run_eval.py）。";
    note.hidden = false;
    return;
  }

  const baseTotal = (report.baseline_input_tokens || 0) + (report.baseline_output_tokens || 0);
  const nowTotal = (report.actual_input_tokens || 0) + (report.actual_output_tokens || 0);
  const llmSaved = clampPct(report.llm_calls_baseline, report.llm_calls_actual);
  const tokenSaved = clampPct(baseTotal, nowTotal);
  const inputSaved = clampPct(report.baseline_input_tokens, report.actual_input_tokens);

  const pctCell = (v) => (v == null ? "--" : `↓ ${(v * 100).toFixed(1)}%`);

  grid.innerHTML = [
    tile("缓存覆盖率", fmtPct(report.cache_coverage_rate), "冷启动", true),
    tile("直接复用率", fmtPct(report.direct_cache_reuse_rate), "冷启动", true),
    tile("LLM 调用节省", pctCell(llmSaved), `${report.llm_calls_baseline} → ${report.llm_calls_actual}`, true),
    tile("Token 节省", pctCell(tokenSaved), `${baseTotal.toLocaleString()} → ${nowTotal.toLocaleString()}`, true),
    tile("动态拦截", report.dynamic_intercepts, "实时问题"),
    tile("输入 Token 节省", pctCell(inputSaved), `${(report.baseline_input_tokens || 0).toLocaleString()} → ${(report.actual_input_tokens || 0).toLocaleString()}`, true)
  ].join("");

  const notes = report.notes || [];
  note.textContent = notes.join(" ");
  note.hidden = notes.length === 0;
}

async function requestChat(question) {
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setStatus("online");
    return await res.json();
  } catch (error) {
    setStatus("offline");
    const hit = FALLBACK_ANSWERS.find((item) => item.test.test(question));
    return hit ? hit.response : {
      answer: "离线预览模式：后端服务未连接。启动服务后，该问题会经过动态拦截 → 两级缓存 → Validator 裁决 → ReAct 检索链路并生成答案。",
      route: "rag", cache_level: null, decision: "rag", sources: [], latency_ms: 290, llm_calls: 1,
      debug: { mode: "offline-demo", error: String(error?.message || error) }
    };
  }
}

async function refreshMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics/eval`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderMetrics(await res.json());
    setStatus("online");
  } catch (_) {
    renderMetrics(FALLBACK_METRICS);
    setStatus("offline");
  }
}

async function submitQuestion(question) {
  addMessage("user", question);
  inputEl.value = "";
  inputEl.disabled = true;

  const bubble = addMessage("assistant", "");
  const typing = document.createElement("span");
  typing.className = "typing";
  typing.innerHTML = "<i></i><i></i><i></i>";
  bubble.appendChild(typing);
  bubble.style.minWidth = "64px";

  const response = await requestChat(question);
  bubble.textContent = response.answer;
  bubble.appendChild(metaTags([
    { label: ROUTE_META[response.route]?.[0] || response.route, cls: "tag-route" },
    response.cache_level ? { label: response.cache_level } : null,
    response.decision ? { label: response.decision } : null,
    { label: `${Number(response.latency_ms || 0).toFixed(2)} ms` },
    { label: `LLM×${response.llm_calls ?? 0}` }
  ]));

  renderDetail(response);
  renderPipeline(response);
  state.lastResponse = response;

  inputEl.disabled = false;
  inputEl.focus();
}

function initPrompts() {
  const row = $("#quickPrompts");
  prompts.forEach((prompt) => {
    const btn = document.createElement("button");
    btn.className = "quick-chip";
    btn.type = "button";
    btn.textContent = prompt;
    btn.addEventListener("click", () => submitQuestion(prompt));
    row.appendChild(btn);
  });
}

$("#chatForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const value = inputEl.value.trim();
  if (value) submitQuestion(value);
});

$("#clearBtn").addEventListener("click", () => {
  messagesEl.innerHTML = "";
  renderDetail(null);
  renderPipeline(null);
  addMessage("assistant", "你好，我是电商智能客服。可以解答商品、售后、物流等问题，试试上面的示例吧～");
});

$("#refreshMetrics").addEventListener("click", refreshMetrics);

initPrompts();
renderMetrics();
addMessage("assistant", "你好，我是电商智能客服。可以解答商品、售后、物流等问题，试试上面的示例吧～");
refreshMetrics();
