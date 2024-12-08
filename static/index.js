const MAX_RETRIES = 3;
const CHAR_LIMIT = 1000;

// Character counter
const userInput = document.getElementById('userInput');
const charCounter = document.getElementById('charCounter');
const sendButton = document.getElementById('sendButton');


// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Get or create session
    fetch('/start')
        .then(response => response.json())
        .then(data => {
            console.log('Session initialized:', data.user_id);
        })
        .catch(error => console.error('Error initializing session:', error));

    const chatContainer = document.getElementById('chatContainer');
    const welcomeMessage = document.getElementById('welcomeMessage');
    
    // Function to check if there are any chat messages
    function updateWelcomeMessage() {
        const messages = chatContainer.getElementsByClassName('message');
        if (messages.length > 0) {
            welcomeMessage.style.display = 'none';
        } else {
            welcomeMessage.style.display = 'block';
        }
    }
    
    // Call initially
    updateWelcomeMessage();
}); 

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

// Auto-focus textarea on any keydown when not already focused
document.addEventListener('keydown', function(event) {
    // Skip if user is pressing a modifier key
    if (event.ctrlKey || event.altKey || event.metaKey) {
        return;
    }
    
    // Skip if current focus is already on textarea
    if (document.activeElement === userInput) {
        return;
    }

    // Skip if user is typing in another input/textarea
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') {
        return;
    }

    userInput.focus();
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
            console.log(error);
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

    // After adding a new message, update welcome message visibility
    updateWelcomeMessage();
}

// Add event listener for Enter key
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
        document.getElementById('userInput').focus();
    }
});

function openRefinePopup() {
    const popup = document.getElementById('refinePopup');
    if (popup) {
        popup.style.display = 'block';
        loadIndices(); // Load indices when the popup is opened
    }
}

async function loadIndices() {
    try {
        const response = await fetch('/indices', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const indices = data.indices || [];
        
        const indexSelect = document.getElementById('indexSelect');
        indexSelect.innerHTML = '<option value="" disabled selected>Choose an index...</option>';
        
        indices.sort().forEach(index => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = index.replace(/-/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
            indexSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading indices:', error);
    }
}

function closeRefinePopup() {
    const popup = document.getElementById('refinePopup');
    if (popup) {
        popup.style.display = 'none';
    }
}

// Ensure this script is correctly linked in your HTML and that there are no JavaScript errors in the console.