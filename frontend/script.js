/* ===========================================================
   Cura AI — front-end behavior, wired to the SAME backend
   configuration as your working chatbot (2.js):
   local -> http://localhost:8000/ask
   prod  -> https://rag-c3793ebc.fastapicloud.dev/ask
=========================================================== */

// ---------- Backend endpoint (identical logic to your working app) ----------
const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const BACKEND_URL = isLocal
  ? "http://localhost:8000/ask"
  : "https://rag-c3793ebc.fastapicloud.dev/ask";

/* =========================================================
   FAQ accordion (index.html)
========================================================= */
document.querySelectorAll(".faq-item").forEach((item) => {
  const q = item.querySelector(".faq-q");
  if (!q) return;
  q.addEventListener("click", () => {
    const isOpen = item.classList.contains("open");
    document.querySelectorAll(".faq-item").forEach((i) => i.classList.remove("open"));
    if (!isOpen) item.classList.add("open");
  });
});

/* =========================================================
   Sign up (signup.html) — no auth backend yet, go straight to chat
========================================================= */
const signupForm = document.getElementById("signup-form");
if (signupForm) {
  signupForm.addEventListener("submit", (e) => {
    e.preventDefault();
    window.location.href = "chat.html";
  });
}

/* =========================================================
   Chat (chat.html) — talks to BACKEND_URL exactly like 2.js
========================================================= */
const chatInner = document.getElementById("chat-inner");
const chatScroll = document.getElementById("chat-scroll");
const composerInput = document.getElementById("composer-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const conversationTitle = document.getElementById("conversation-title");

const EMPTY_STATE_HTML = `
  <div id="chat-empty" style="min-height:60vh;display:flex;align-items:center;justify-content:center;text-align:center;color:var(--ink-600);padding:30px;">
    <div>
      <div style="font-size:28px;font-weight:700;margin-bottom:10px;">Cura AI</div>
      <div style="font-size:14px;">Ask a clinical question or describe a case.</div>
    </div>
  </div>`;

function removeEmptyState() {
  const empty = document.getElementById("chat-empty");
  if (empty) empty.remove();
}

function appendUserBubble(text) {
  removeEmptyState();
  const row = document.createElement("div");
  row.className = "msg-row";
  row.innerHTML = `<div class="avatar avatar-user">U</div><div class="msg-text"></div>`;
  row.querySelector(".msg-text").textContent = text;
  chatInner.appendChild(row);
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function appendAiBubble(html) {
  const row = document.createElement("div");
  row.className = "msg-row msg-ai";
  row.innerHTML = `
    <div class="avatar avatar-ai">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>
    </div>
    <div class="msg-text"></div>`;
  row.querySelector(".msg-text").innerHTML = html;
  chatInner.appendChild(row);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return row;
}

async function handleSend() {
  if (!composerInput) return;
  const query = composerInput.value.trim();
  if (!query) return;

  appendUserBubble(query);
  composerInput.value = "";
  composerInput.style.height = "auto";

  if (conversationTitle && conversationTitle.textContent.trim() === "New conversation") {
    conversationTitle.textContent = query.length > 42 ? query.slice(0, 42) + "…" : query;
  }

  const loadingRow = appendAiBubble(
    `<span style="font-style:italic;color:var(--ink-400);">Analyzing medical literature…</span>`
  );

  try {
    // Same request shape as your working 2.js
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, patient_context: "" }),
    });

    if (!response.ok) throw new Error(`Backend error: ${response.status}`);

    const data = await response.json();
    loadingRow.remove();
    appendAiBubble(data.answer);
  } catch (err) {
    loadingRow.remove();
    appendAiBubble(
      `<span style="color:#be123c;">⚠️ Unable to reach Cura AI's backend at <code>${BACKEND_URL}</code>. Check that it's running.</span>`
    );
  }
}

if (composerInput) {
  composerInput.addEventListener("input", () => {
    composerInput.style.height = "auto";
    composerInput.style.height = Math.min(composerInput.scrollHeight, 160) + "px";
  });
  composerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
}
if (sendBtn) sendBtn.addEventListener("click", handleSend);

if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    if (chatInner) chatInner.innerHTML = EMPTY_STATE_HTML;
    if (conversationTitle) conversationTitle.textContent = "New conversation";
  });
}

// Sidebar toggle on small screens
function syncSidebarToggle() {
  if (!sidebarToggle) return;
  sidebarToggle.style.display = window.innerWidth <= 820 ? "grid" : "none";
}
if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  window.addEventListener("resize", syncSidebarToggle);
  syncSidebarToggle();
}


