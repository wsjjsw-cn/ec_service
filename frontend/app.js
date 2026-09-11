const API_BASE = window.location.origin.startsWith("file")
  ? "http://127.0.0.1:8000"
  : window.location.origin;

// 开场建议
const suggestions = [
  "七天无理由退货需要满足什么条件？",
  "退款一般多长时间到账",
  "AeroPods Pro 2 现在还有库存吗？",
  "保修范围包括哪些内容？"
];

// 路由 → 面向用户的简短说明（用于回答下方的极小字注脚）
const CAPTIONS = {
  dynamic: "实时数据",
  cache: "缓存命中",
  rag: "知识库检索",
  identity: "快捷回复",
  chitchat: "快捷回复",
  ood: "引导回复"
};

// 离线演示回答：连接不上后端时使用，保证静态预览仍可交互
const FALLBACK_ANSWERS = [
  { test: /退货|七天/, response: {
      answer: "七天无理由退货需在签收次日起 7 天内发起，商品、配件、赠品、包装和发票保持完好、不影响二次销售。定制、生鲜，以及拆封后影响安全或卫生的商品不支持无理由退货。",
      route: "cache", cache_level: "L1", latency_ms: 38 } },
  { test: /退款|到账/, response: {
      answer: "退款在售后审核通过后原路退回。银行卡通常 1~5 个工作日到账，支付宝与微信通常 24 小时内到账，具体以支付机构的处理时间为准。",
      route: "cache", cache_level: "L1", latency_ms: 41 } },
  { test: /库存|有货|AeroPods/i, response: {
      answer: "AeroPods Pro 2 蓝牙耳机当前有货，库存 12 件，数据更新于 2026-08-01 10:00。下单前建议以商品页的实时库存为准。",
      route: "dynamic", latency_ms: 17 } },
  { test: /保修|维修|三包/, response: {
      answer: "家电、数码等商品按国家三包和品牌政策提供保修。保修期内非人为的质量问题可免费维修；人为损坏、进水、私自拆修不在免费保修范围内。",
      route: "rag", latency_ms: 52 } },
  { test: /你是谁|介绍.*你|介绍下你/, response: {
      answer: "我是小智，电商智能客服。商品咨询、订单查询、退换货、物流与售后政策，都可以问我。",
      route: "identity", latency_ms: 2 } },
  { test: /^(你好|您好|hi|hello)/i, response: {
      answer: "您好，我是小智。请问需要了解点什么？",
      route: "chitchat", latency_ms: 2 } },
  { test: /谢谢|感谢|再见|拜拜/, response: {
      answer: "不客气，有需要随时找我。",
      route: "chitchat", latency_ms: 2 } },
  { test: /笑话|天气|新闻|游戏|电影/, response: {
      answer: "我主要负责商品与售后相关的问题，这个我不太擅长。如果您有购物、订单或售后方面的疑问，我很乐意帮忙。",
      route: "ood", latency_ms: 3 } }
];

const thread = document.getElementById("thread");
const form = document.getElementById("chatForm");
const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const presence = document.getElementById("presence");

function setPresence(mode) {
  presence.classList.remove("is-online", "is-pending");
  presence.classList.add(mode === "online" ? "is-online" : "is-pending");
  presence.querySelector(".presence-text").textContent = mode === "online" ? "在线" : "离线演示";
}

function showStarter() {
  thread.innerHTML = "";
  const starter = document.createElement("div");
  starter.className = "starter";

  const kicker = document.createElement("p");
  kicker.className = "starter-kicker";
  kicker.textContent = "Customer Care";

  const title = document.createElement("h1");
  title.className = "starter-title";
  title.textContent = "有什么可以帮您？";

  const sub = document.createElement("p");
  sub.className = "starter-sub";
  sub.textContent = "商品咨询、订单查询、退换货与物流问题，都可以直接问我。";

  const list = document.createElement("div");
  list.className = "starter-list";
  suggestions.forEach((q) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "starter-chip";
    chip.textContent = q;
    chip.addEventListener("click", () => submit(q));
    list.appendChild(chip);
  });

  starter.append(kicker, title, sub, list);
  thread.appendChild(starter);
}

function addMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  msg.appendChild(body);
  thread.appendChild(msg);
  thread.scrollTop = thread.scrollHeight;
  return { msg, body };
}

function addCaption(msgEl, route, latency) {
  const label = CAPTIONS[route];
  if (!label) return;
  const cap = document.createElement("div");
  cap.className = "caption";
  cap.textContent = `${label} · ${Number(latency || 0).toFixed(0)} ms`;
  msgEl.appendChild(cap);
}

async function askBackend(question) {
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setPresence("online");
    return await res.json();
  } catch (_) {
    setPresence("offline");
    const hit = FALLBACK_ANSWERS.find((item) => item.test.test(question));
    return hit ? hit.response : {
      answer: "暂时无法连接到服务，请稍后再试。",
      route: "rag", latency_ms: 0
    };
  }
}

async function submit(question) {
  const starter = thread.querySelector(".starter");
  if (starter) starter.remove();

  addMessage("user", question);
  input.value = "";
  setBusy(true);

  const { msg, body } = addMessage("assistant", "");
  const typing = document.createElement("span");
  typing.className = "typing";
  typing.innerHTML = "<i></i><i></i><i></i>";
  body.appendChild(typing);

  const response = await askBackend(question);
  body.textContent = response.answer;
  addCaption(msg, response.route, response.latency_ms);

  setBusy(false);
  input.focus();
}

function setBusy(busy) {
  input.disabled = busy;
  sendBtn.disabled = busy;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = input.value.trim();
  if (value) submit(value);
});

document.getElementById("resetBtn").addEventListener("click", () => {
  showStarter();
  setBusy(false);
  input.focus();
});

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    setPresence(res.ok ? "online" : "offline");
  } catch (_) {
    setPresence("offline");
  }
}

showStarter();
checkHealth();
