const API_BASE = window.location.origin.startsWith("file") ? "http://127.0.0.1:8000" : window.location.origin;

const prompts = [
  "你是谁",
  "讲个笑话",
  "七天无理由退货需要满足什么条件？",
  "AeroPods Pro 2 现在还有库存吗？",
  "退款一般多长时间到账",
  "如何维修商品",
  "谢谢"
];

// 后端未连接时的占位提示。这里刻意不放任何具体指标数值 —— 离线预览下展示
// 「缓存覆盖 42.31% / 吞吐 1.36x」这类数字会让人误以为是实测结果。
const fallbackMetrics = {
  offline: true,
  dynamic_intercepts: null,
  cache_coverage_rate: null,
  direct_cache_reuse_rate: null,
  throughput_multiplier: null,
  latency_reduction_rate: null,
  total_llm_saved: null,
  cost_saving_rate: null
};

const fallbackAnswers = [
  {
    test: /^(你|您)是(谁|啥|什么|哪位|做什么)/,
    response: {
      answer: "您好，我是智能电商客服助手。我可以帮您解答商品咨询、售后政策、订单查询、物流追踪等问题。请问有什么可以帮您？",
      route: "identity",
      cache_level: null,
      decision: null,
      sources: [],
      latency_ms: 1.2,
      llm_calls: 0,
      debug: { intent: "identity", mode: "offline-demo" }
    }
  },
  {
    test: /^(你好|您好|hello|hi)/i,
    response: {
      answer: "您好！我是智能电商客服助手，请问有什么可以帮您？",
      route: "chitchat",
      cache_level: null,
      decision: null,
      sources: [],
      latency_ms: 1.0,
      llm_calls: 0,
      debug: { intent: "chitchat", mode: "offline-demo" }
    }
  },
  {
    test: /^(谢谢|感谢|再见|拜拜|bye)/i,
    response: {
      answer: "不客气，很高兴能帮到您！如果还有其他问题，随时可以问我。",
      route: "chitchat",
      cache_level: null,
      decision: null,
      sources: [],
      latency_ms: 1.0,
      llm_calls: 0,
      debug: { intent: "chitchat", mode: "offline-demo" }
    }
  },
  {
    test: /讲个笑话|有什么游戏|电影推荐/,
    response: {
      answer: "抱歉，我主要是电商客服助手，只能解答商品、订单、售后、物流等购物相关问题。您的问题不在我的服务范围内，建议您咨询相关客服或查看帮助中心。",
      route: "ood",
      cache_level: null,
      decision: null,
      sources: [],
      latency_ms: 2.3,
      llm_calls: 0,
      debug: { intent: "out_of_domain", mode: "offline-demo" }
    }
  },
  {
    test: /库存|有货|AeroPods/i,
    response: {
      answer: "AeroPods Pro 2 蓝牙耳机 当前有货，库存为 12 件。库存数据更新时间：2026-08-01T10:00:00+08:00，下单前建议以商品页实时库存为准。",
      route: "dynamic",
      cache_level: null,
      decision: null,
      sources: [],
      latency_ms: 18.4,
      llm_calls: 0,
      debug: { guardrail: "stock", mode: "offline-demo" }
    }
  },
  {
    test: /退款|到账/,
    response: {
      answer: "退款在售后审核通过后原路退回。银行卡通常1到5个工作日到账，支付宝和微信通常24小时内到账，具体以支付机构处理时间为准。",
      route: "cache",
      cache_level: "L1",
      decision: "reuse",
      sources: ["faq_refund_time"],
      latency_ms: 42.8,
      llm_calls: 0,
      debug: { reason: "l1_intent_shortcut", score: 0.98, mode: "offline-demo" }
    }
  },
  {
    test: /退货|七天/,
    response: {
      answer: "七天无理由退货需在签收次日起7天内发起，商品、配件、赠品、包装和发票保持完好，不影响二次销售。定制、生鲜、拆封后影响安全或卫生的商品不支持无理由退货。",
      route: "cache",
      cache_level: "L1",
      decision: "reuse",
      sources: ["faq_return_7d"],
      latency_ms: 39.6,
      llm_calls: 0,
      debug: { reason: "l1_exact_normalized", score: 1, mode: "offline-demo" }
    }
  },
  {
    test: /如何维修|保修政策/i,
    response: {
      answer: "家电、数码等商品按国家三包和品牌政策提供保修。保修期内出现质量问题可申请免费维修；人为损坏、进水、私自拆修不在免费保修范围内。",
      route: "rag",
      cache_level: null,
      decision: "rag",
      sources: ["kb_warranty", "kb_repair"],
      latency_ms: 45.2,
      llm_calls: 1,
      debug: { reason: "full_rag_retrieval", mode: "offline-demo" }
    }
  }
];

const state = { online: false, lastResponse: null };

const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
const input = $("#questionInput");

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setStatus(online) {
  state.online = online;
  const node = $("#serviceStatus");
  node.classList.toggle("online", online);
  node.classList.toggle("offline", !online);
  node.querySelector("span:last-child").textContent = online ? "后端在线" : "离线预览";
}

function addMessage(role, text, meta = "") {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  if (meta) {
    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";
    metaNode.textContent = meta;
    bubble.appendChild(metaNode);
  }
  item.appendChild(bubble);
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function renderRoute(response) {
  const routeMap = {
    dynamic: ["动态拦截", "实时风险问题已跳过静态缓存"],
    identity: ["身份识别", "用户询问身份，返回介绍"],
    chitchat: ["闲聊对话", "问候/感谢/告别等礼貌回应"],
    ood: ["领域外问题", "问题超出电商客服范围，已返回友好提示"],
    cache: ["缓存命中", "答案来自 L1/L2 缓存或补充生成"],
    rag: ["完整 RAG", "缓存收益不足，进入混合检索与生成"]
  };
  const [title, desc] = routeMap[response.route] || ["未知链路", "未识别响应"];
  $("#routeCard").innerHTML = `<span class="route-label">${response.route || "-"}</span><strong>${title}</strong><p>${desc}</p>`;
  $("#detailGrid").innerHTML = `
    <div><dt>缓存层</dt><dd>${response.cache_level || "-"}</dd></div>
    <div><dt>决策</dt><dd>${response.decision || "-"}</dd></div>
    <div><dt>延迟</dt><dd>${Number(response.latency_ms || 0).toFixed(2)} ms</dd></div>
    <div><dt>LLM 调用</dt><dd>${response.llm_calls ?? 0}</dd></div>
  `;
  $("#traceBox").textContent = JSON.stringify({ sources: response.sources, debug: response.debug }, null, 2);
}

function renderMetrics(report = fallbackMetrics) {
  const dash = () => "--";
  if (report.offline) {
    $("#metricsGrid").innerHTML = `
      <div class="metric"><span>动态拦截</span><strong>${dash()}</strong></div>
      <div class="metric"><span>缓存覆盖</span><strong>${dash()}</strong></div>
      <div class="metric"><span>直接复用</span><strong>${dash()}</strong></div>
      <div class="metric"><span>吞吐提升</span><strong>${dash()}</strong></div>
    `;
    const note = $("#metricsNote");
    if (note) {
      note.textContent = "未连接后端：指标需运行 python scripts/run_eval.py 或启动服务后由 /metrics/eval 实测生成。";
      note.style.display = "block";
    }
    return;
  }
  $("#metricsGrid").innerHTML = `
    <div class="metric"><span>动态拦截</span><strong>${report.dynamic_intercepts}</strong></div>
    <div class="metric"><span>缓存覆盖</span><strong>${formatPercent(report.cache_coverage_rate)}</strong></div>
    <div class="metric"><span>直接复用</span><strong>${formatPercent(report.direct_cache_reuse_rate)}</strong></div>
    <div class="metric"><span>吞吐提升</span><strong>${Number(report.throughput_multiplier).toFixed(2)}x</strong></div>
  `;
  const note = $("#metricsNote");
  if (note) {
    const notes = report.notes || [];
    note.textContent = notes.join(" ");
    note.style.display = notes.length ? "block" : "none";
  }
}

async function requestChat(question) {
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setStatus(true);
    return await res.json();
  } catch (error) {
    setStatus(false);
    const hit = fallbackAnswers.find((item) => item.test.test(question));
    return hit ? hit.response : {
      answer: "离线预览模式：后端服务未连接。启动 uvicorn 后可查看真实 RAG/缓存链路；当前问题会在后端进入混合检索并生成答案。",
      route: "rag",
      cache_level: null,
      decision: "rag",
      sources: [],
      latency_ms: 290,
      llm_calls: 1,
      debug: { mode: "offline-demo", error: String(error.message || error) }
    };
  }
}

async function refreshMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics/eval`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderMetrics(await res.json());
    setStatus(true);
  } catch (_) {
    renderMetrics(fallbackMetrics);
    setStatus(false);
  }
}

async function submitQuestion(question) {
  addMessage("user", question);
  input.value = "";
  input.disabled = true;
  const pending = "正在经过动态拦截、缓存和检索链路...";
  addMessage("assistant", pending);
  const pendingNode = messages.lastElementChild.querySelector(".bubble");
  const response = await requestChat(question);
  pendingNode.textContent = response.answer;
  const metaNode = document.createElement("div");
  metaNode.className = "message-meta";
  metaNode.textContent = `${response.route}${response.cache_level ? ` · ${response.cache_level}` : ""} · ${Number(response.latency_ms || 0).toFixed(2)} ms`;
  pendingNode.appendChild(metaNode);
  renderRoute(response);
  input.disabled = false;
  input.focus();
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
  const value = input.value.trim();
  if (value) submitQuestion(value);
});

$("#clearBtn").addEventListener("click", () => {
  messages.innerHTML = "";
  $("#traceBox").textContent = "{}";
  addMessage("assistant", "你好，我是电商智能客服。可以解答商品、售后、物流等问题。试试上方的示例吧！");
});

$("#refreshMetrics").addEventListener("click", refreshMetrics);

initPrompts();
renderMetrics();
addMessage("assistant", "你好，我是电商智能客服。可以解答商品、售后、物流等问题。试试上方的示例吧！");
refreshMetrics();
