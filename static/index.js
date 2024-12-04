const MAX_RETRIES = 3;
const CHAR_LIMIT = 1000;

// Character counter
const userInput = document.getElementById('userInput');
const charCounter = document.getElementById('charCounter');
const sendButton = document.getElementById('sendButton');

userInput.addEventListener('input', function() {
    const length = this.value.length;
    charCounter.textContent = `${length}/${CHAR_LIMIT}`;
    
    if (length >= CHAR_LIMIT) {
        charCounter.classList.add('at-limit');
        charCounter.classList.remove('near-limit');
    } else if (length >= CHAR_LIMIT * 0.9) {
        charCounter.classList.add('near-limit');
        charCounter.classList.remove('at-limit');
    } else {
        charCounter.classList.remove('near-limit', 'at-limit');
    }
});

function formatTimestamp() {
    return new Date().toLocaleTimeString();
}

async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy text:', err);
        return false;
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    const chatContainer = document.getElementById('chatContainer');
    
    // Add user message with timestamp
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user-message';
    userMessageDiv.innerHTML = `
        <div class="timestamp">${formatTimestamp()}</div>
        <div class="message-content">${message}</div>
    `;
    chatContainer.appendChild(userMessageDiv);
    userInput.value = '';
    charCounter.textContent = `0/${CHAR_LIMIT}`;

    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = `
        <div class="loading-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    chatContainer.appendChild(loadingDiv);

    // Disable input while processing
    userInput.disabled = true;
    sendButton.disabled = true;

    let retries = 0;
    while (retries < MAX_RETRIES) {
        try {
            const response = await fetch('/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            loadingDiv.remove();

            let responseHTML = '';
            if (data.type === 'text') {
                responseHTML = `
                    <div class="message response-message">
                        <div class="response-content">
                            <div class="timestamp">${formatTimestamp()}</div>
                            <div class="response-type">Type: ${data.type}</div>
                            <button class="copy-button" onclick="copyToClipboard(this.parentElement.textContent).then(success => this.textContent = success ? 'Copied!' : 'Failed')">Copy</button>
                            <div class="topic">${data.topic}</div>
                            <div class="paragraph">${data.paragraph}</div>
                            ${data.body.map(item => `
                                <div class="sub_topics">${item.sub_topics}</div>
                                <div class="paragraph">${item.content}</div>
                            `).join('')}
                            <div class="summary"><span class="label">Summary:</span>${data.summary}</div>
                            <div class="caption"><span class="label">Caption:</span>${data.caption}</div>
                            <div class="hashtags">
                                ${data.hashtags.map(tag => `<span>${tag}</span>`).join('')}
                            </div>
                        </div>
                    </div>
                `;
            } else if (data.type === 'carousel') {
                responseHTML = `
                    <div class="message response-message">
                        <div class="response-content">
                            <div class="response-type">Type: ${data.type}</div>
                            <div class="topic">${data.topic || 'No Topic'}</div>
                            <div class="paragraph">${data.paragraph || ''}</div>
                            ${Array.isArray(data.body) ? data.body.map(item => `
                                <div class="sub_topics">${item.sub_topics || ''}</div>
                                <div class="paragraph">${item.content || ''}</div>
                            `).join('') : ''}
                            <div class="summary"><span class="label">Summary:</span>${data.summary || ''}</div>
                            <div class="caption"><span class="label">Caption:</span>${data.caption || ''}</div>
                            <div class="hashtags">
                                ${Array.isArray(data.hashtags) ? data.hashtags.map(tag => `<span>${tag}</span>`).join('') : ''}
                            </div>
                        </div>
                    </div>
                `;
            } else if (data.type === 'podcast') {
                responseHTML = `
                    <div class="message response-message">
                        <div class="response-content">
                            <div class="response-type">Type: ${data.type}</div>
                            <div class="podcast-response">${data.answer || 'No response available'}</div>
                        </div>
                    </div>
                `;
            } else {
                responseHTML = `
                    <div class="message response-message">
                        <div class="response-content">
                            <div class="error-message">${data.answer || 'No response available'}</div>
                        </div>
                    </div>
                `;
            }

            chatContainer.insertAdjacentHTML('beforeend', responseHTML);
            break;
        } catch (error) {
            retries++;
            if (retries === MAX_RETRIES) {
                loadingDiv.remove();
                const errorHTML = `
                    <div class="message response-message">
                        <div class="timestamp">${formatTimestamp()}</div>
                        <div class="error-message">Failed to process your request after ${MAX_RETRIES} attempts. Please try again later.</div>
                    </div>
                `;
                chatContainer.insertAdjacentHTML('beforeend', errorHTML);
            }
            await new Promise(resolve => setTimeout(resolve, 1000 * retries)); // Exponential backoff
        } finally {
            userInput.disabled = false;
            sendButton.disabled = false;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
}

// Add event listener for Enter key
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});