const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');

// Change this to your live FastAPI Cloud URL when deployed
const BACKEND_URL = 'https://rag-c3793ebc.fastapicloud.dev/ask';

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // Append user message to chat UI
    appendMessage(query, 'user');
    userInput.value = '';

    // Show loading indicator
    const loadingId = appendMessage('Analyzing medical literature...', 'assistant loading');

    try {
        const response = await fetch(BACKEND_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });

        // Remove loading message
        document.getElementById(loadingId).remove();

        if (response.ok) {
            const data = await response.json();
            appendMessage(data.answer, 'assistant');
        } else {
            const errorData = await response.json();
            appendMessage(`⚠️ **Server Error:** ${errorData.detail || 'Unknown error'}`, 'assistant');
        }
    } catch (error) {
        document.getElementById(loadingId).remove();
        appendMessage('⚠️ **Connection Error:** Could not reach the FastAPI backend server.', 'assistant');
    }
});

function appendMessage(text, senderClass) {
    const messageDiv = document.createElement('div');
    const messageId = 'msg-' + Date.now();
    messageDiv.id = messageId;
    messageDiv.className = `message ${senderClass}`;

    // Render HTML tags instead of printing raw text tags
    messageDiv.innerHTML = text;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageId;
}
