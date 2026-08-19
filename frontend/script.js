const BACKEND_URL = "https://rag-c3793ebc.fastapicloud.dev/ask";

const chatMessages = document.getElementById("chatMessages");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");

async function handleSendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;

    // 1. Append User Message
    appendMessage(query, "user");
    queryInput.value = "";
    queryInput.focus();

    // 2. Scroll to bottom
    scrollToBottom();

    // 3. Optional loading placeholder
    const loadingId = appendMessage("Analyzing clinical sources...", "assistant", true);

    try {
        const response = await fetch(BACKEND_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();

        // Remove loading message
        document.getElementById(loadingId).remove();

        // Append assistant HTML answer
        appendHtmlMessage(data.answer, "assistant");

    } catch (error) {
        document.getElementById(loadingId).remove();
        appendMessage("⚠️ Error connecting to the clinical backend. Please make sure FastAPI is running.", "assistant");
        console.error("Fetch error:", error);
    }
}

function appendMessage(text, sender, isLoading = false) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    if (isLoading) {
        msgDiv.id = "loading-msg";
        msgDiv.style.fontStyle = "italic";
        msgDiv.style.color = "var(--text-muted)";
    }
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv.id;
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

// Event Listeners
sendBtn.addEventListener("click", handleSendMessage);
queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        handleSendMessage();
    }
});
