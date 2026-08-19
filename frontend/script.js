const BACKEND_URL = "https://rag-c3793ebc.fastapicloud.dev/ask";

const chatMessages = document.getElementById("chatMessages");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");

async function handleSendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;

    appendMessage(query, "user");
    queryInput.value = "";
    queryInput.focus();

    // Use a unique ID for the loading message
    const loadingId = "load-" + Date.now();
    const loadingMsg = document.createElement("div");
    loadingMsg.id = loadingId;
    loadingMsg.classList.add("message", "assistant");
    loadingMsg.style.fontStyle = "italic";
    loadingMsg.textContent = "Analyzing medical literature...";
    chatMessages.appendChild(loadingMsg);
    scrollToBottom();

    try {
        const response = await fetch(BACKEND_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, patient_context: "" })
        });

        const data = await response.json();

        // Remove specific loading message
        const el = document.getElementById(loadingId);
        if (el) el.remove();

        appendHtmlMessage(data.answer, "assistant");

    } catch (error) {
        const el = document.getElementById(loadingId);
        if (el) el.remove();
        appendMessage("⚠️ Error: Unable to connect to backend.", "assistant");
    }
}

function appendMessage(text, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function appendHtmlMessage(htmlContent, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.innerHTML = htmlContent;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

sendBtn.addEventListener("click", handleSendMessage);
queryInput.addEventListener("keydown", (e) => { if (e.key === "Enter") handleSendMessage(); });
