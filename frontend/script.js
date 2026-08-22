/* ===========================================================
   Cura AI — front-end behavior, wired to the backend
=========================================================== */

// ---------- Backend endpoint configuration ----------
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
   Sign up (signup.html) — go straight to chat
========================================================= */
const signupForm = document.getElementById("signup-form");
if (signupForm) {
  signupForm.addEventListener("submit", (e) => {
    e.preventDefault();
    window.location.href = "chat.html";
  });
}

/* =========================================================
   Chat (chat.html) — talks to BACKEND_URL
========================================================= */
const chatInner = document.getElementById("chat-inner");
const chatScroll = document.getElementById("chat-scroll");
const composerInput = document.getElementById("composer-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat");
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
        `<span style="color:#be123c;">⚠️ Unable to connect to Cura AI at the moment. Please try again in a few moments.</span>`
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

/* =========================================================
   Mobile Sidebar Toggle Wiring
========================================================= */
document.addEventListener("DOMContentLoaded", () => {
  const sidebarElement = document.querySelector(".app-sidebar");
  const toggleBtn = document.querySelector("#sidebar-toggle") || document.querySelector(".app-topbar .icon-btn");

  if (toggleBtn && sidebarElement) {
    toggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      sidebarElement.classList.toggle("open");
    });

    // Close sidebar when clicking outside on mobile views
    document.addEventListener("click", (e) => {
      if (window.innerWidth <= 768 && sidebarElement.classList.contains("open")) {
        if (!sidebarElement.contains(e.target) && !toggleBtn.contains(e.target)) {
          sidebarElement.classList.remove("open");
        }
      }
    });
  }
});
