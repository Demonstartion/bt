class ChatApp {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatForm = document.getElementById('chatForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.modelSelect = document.getElementById('modelSelect');
        this.loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        this.chatContainer = document.getElementById('chatContainer');
    }

    init() {
        this.bindEvents();
        this.loadChatHistory();
        this.messageInput.focus();
    }

    bindEvents() {
        // Chat form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // New chat button
        this.newChatBtn.addEventListener('click', () => {
            this.startNewChat();
        });

        // Enter key handling
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    async loadChatHistory() {
        try {
            const response = await fetch('/get_chat_history');
            const data = await response.json();
            
            if (data.history && data.history.length > 0) {
                this.clearWelcomeMessage();
                data.history.forEach(exchange => {
                    this.addMessage(exchange.user, 'user');
                    this.addMessage(exchange.bot, 'bot', exchange.model);
                });
                this.scrollToBottom();
            }
            
            // Set selected model
            if (data.selected_model) {
                this.modelSelect.value = data.selected_model;
            }
            
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        const selectedModel = this.modelSelect.value;

        // Clear welcome message if it exists
        this.clearWelcomeMessage();

        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input and disable form
        this.messageInput.value = '';
        this.setLoading(true);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    model: selectedModel
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Add bot response to chat
                this.addMessage(data.response, 'bot', data.model_used);
            } else {
                this.addMessage(`Error: ${data.error}`, 'bot', 'error');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, there was an error processing your message. Please try again.', 'bot', 'error');
        } finally {
            this.setLoading(false);
            this.messageInput.focus();
        }
    }

    addMessage(text, sender, model = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        if (sender === 'user') {
            messageDiv.innerHTML = `
                <div class="d-flex justify-content-end mb-3">
                    <div class="message-bubble user-message">
                        <div class="message-content">${this.escapeHtml(text)}</div>
                        <small class="text-muted">${timestamp}</small>
                    </div>
                </div>
            `;
        } else {
            const modelBadge = model && model !== 'error' ? 
                `<span class="badge bg-secondary me-2">${this.getModelDisplayName(model)}</span>` : '';
            
            messageDiv.innerHTML = `
                <div class="d-flex justify-content-start mb-3">
                    <div class="message-bubble bot-message">
                        <div class="message-header mb-1">
                            <i data-feather="cpu" style="width: 16px; height: 16px;" class="me-1"></i>
                            ${modelBadge}
                            <small class="text-muted">${timestamp}</small>
                        </div>
                        <div class="message-content">${this.formatBotMessage(text)}</div>
                    </div>
                </div>
            `;
        }

        this.chatMessages.appendChild(messageDiv);
        
        // Re-initialize feather icons for new content
        feather.replace();
        
        this.scrollToBottom();
    }

    formatBotMessage(text) {
        // Convert line breaks to HTML
        let formatted = this.escapeHtml(text);
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Format lists
        formatted = formatted.replace(/^• (.+)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        return formatted;
    }

    getModelDisplayName(model) {
        const modelNames = {
            'gemini-2.0-flash-exp': 'Gemini 2.0 Flash',
            'gpt-4o-mini': 'GPT-4o Mini',
            'gpt-4o': 'GPT-4o',
            'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet'
        };
        return modelNames[model] || model;
    }

    clearWelcomeMessage() {
        const welcomeMsg = this.chatMessages.querySelector('.text-center.text-muted');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }
    }

    async startNewChat() {
        if (confirm('Are you sure you want to start a new chat? This will clear your current conversation.')) {
            try {
                const response = await fetch('/new_chat', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    this.chatMessages.innerHTML = `
                        <div class="text-center text-muted py-5">
                            <i data-feather="message-circle" style="width: 48px; height: 48px;"></i>
                            <h5 class="mt-3">Start a conversation</h5>
                            <p>Ask me anything! I have access to current information through internet search.</p>
                        </div>
                    `;
                    feather.replace();
                    this.messageInput.focus();
                } else {
                    alert('Error starting new chat. Please try again.');
                }
            } catch (error) {
                console.error('Error starting new chat:', error);
                alert('Error starting new chat. Please try again.');
            }
        }
    }

    setLoading(loading) {
        if (loading) {
            this.loadingModal.show();
            this.sendBtn.disabled = true;
            this.messageInput.disabled = true;
            this.newChatBtn.disabled = true;
        } else {
            this.loadingModal.hide();
            this.sendBtn.disabled = false;
            this.messageInput.disabled = false;
            this.newChatBtn.disabled = false;
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        }, 100);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
