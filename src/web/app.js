// ========== STATE MANAGEMENT ==========
const state = {
  busy: false,
  currentChatId: null,
  chats: {},
};

// ========== DOM ELEMENTS ==========
const app = {
  sidebar: document.getElementById("sidebar"),
  sidebarOpenBtn: document.getElementById("sidebar-open"),
  sidebarCloseBtn: document.getElementById("sidebar-close"),
  chatHistory: document.getElementById("chat-history"),
  newChatBtn: document.getElementById("new-chat-btn"),
  clearHistoryBtn: document.getElementById("clear-history-btn"),
  promptInput: document.getElementById("prompt"),
  sendBtn: document.getElementById("send-btn"),
  messagesContainer: document.getElementById("messages"),
  emptyState: document.getElementById("empty-state"),
  suggestedPromptsContainer: document.getElementById("suggested-prompts"),
};

// ========== STORAGE & INITIALIZATION ==========
function loadChats() {
  const stored = localStorage.getItem("splm_chats");
  return stored ? JSON.parse(stored) : {};
}

function saveChats() {
  localStorage.setItem("splm_chats", JSON.stringify(state.chats));
}

function createNewChat() {
  const chatId = Date.now().toString();
  state.chats[chatId] = {
    id: chatId,
    title: "New Chat",
    messages: [],
    createdAt: new Date().toISOString(),
  };
  saveChats();
  return chatId;
}

function selectChat(chatId) {
  state.currentChatId = chatId;
  renderMessages();
  updateChatHistoryUI();
  app.messagesContainer.scrollTop = app.messagesContainer.scrollHeight;
  app.promptInput.focus();
}

// ========== UI RENDERING ==========
function formatTime(timestamp) {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  } else if (date.toDateString() === yesterday.toDateString()) {
    return "Yesterday";
  } else {
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  }
}

function updateChatHistoryUI() {
  if (Object.keys(state.chats).length === 0) {
    app.chatHistory.innerHTML =
      '<span class="empty-state">No chat history</span>';
    return;
  }

  app.chatHistory.innerHTML = "";
  const sortedChats = Object.values(state.chats).sort(
    (a, b) => new Date(b.createdAt) - new Date(a.createdAt),
  );

  sortedChats.forEach((chat) => {
    const isActive = chat.id === state.currentChatId;
    const li = document.createElement("button");
    li.className = `chat-history-item ${isActive ? "active" : ""}`;
    li.setAttribute("aria-pressed", isActive);

    const title =
      chat.messages.length > 0
        ? chat.messages[0].text.substring(0, 40).trim()
        : chat.title;

    const titleSpan = document.createElement("span");
    titleSpan.textContent = title || "New Chat";
    titleSpan.style.flex = "1";
    titleSpan.style.textAlign = "left";
    titleSpan.style.overflow = "hidden";
    titleSpan.style.textOverflow = "ellipsis";

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "chat-history-item-delete";
    deleteBtn.setAttribute("aria-label", "Delete chat");
    deleteBtn.innerHTML = `<svg class="icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14zM10 11v6M14 11v6"/>
    </svg>`;

    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (confirm("Delete this chat?")) {
        delete state.chats[chat.id];
        saveChats();
        if (state.currentChatId === chat.id) {
          const remaining = Object.keys(state.chats);
          state.currentChatId = remaining.length > 0 ? remaining[0] : null;
        }
        updateChatHistoryUI();
        if (state.currentChatId) {
          selectChat(state.currentChatId);
        } else {
          app.messagesContainer.innerHTML = "";
          showEmptyState();
        }
      }
    });

    li.appendChild(titleSpan);
    li.appendChild(deleteBtn);
    li.addEventListener("click", () => selectChat(chat.id));
    app.chatHistory.appendChild(li);
  });
}

function renderMessages() {
  app.messagesContainer.innerHTML = "";

  if (!state.currentChatId || !state.chats[state.currentChatId]?.messages) {
    showEmptyState();
    return;
  }

  const messages = state.chats[state.currentChatId].messages;
  if (messages.length === 0) {
    showEmptyState();
    return;
  }

  app.emptyState?.remove();

  messages.forEach((msg, index) => {
    const messageEl = document.createElement("div");
    messageEl.className = `message ${msg.role}`;
    messageEl.setAttribute("role", msg.role === "bot" ? "article" : "status");

    const contentEl = document.createElement("div");
    contentEl.className = "message-content";

    const bubbleEl = document.createElement("div");
    bubbleEl.className = "message-bubble";
    bubbleEl.textContent = msg.text;
    if (msg.thinking) {
      bubbleEl.classList.add("thinking");
    }

    contentEl.appendChild(bubbleEl);

    // Add timestamp and actions for bot messages
    if (msg.role === "bot") {
      const timestampEl = document.createElement("div");
      timestampEl.className = "message-timestamp";
      timestampEl.textContent = msg.timestamp ? formatTime(msg.timestamp) : "";

      const actionsEl = document.createElement("div");
      actionsEl.className = "message-actions";

      const copyBtn = document.createElement("button");
      copyBtn.className = "message-action-btn copy";
      copyBtn.setAttribute("aria-label", "Copy message");
      copyBtn.innerHTML = `<svg class="icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2M9 9h6M9 13h6"/>
      </svg>`;
      copyBtn.addEventListener("click", () =>
        copyToClipboard(msg.text, copyBtn),
      );

      const speakBtn = document.createElement("button");
      speakBtn.className = "message-action-btn";
      speakBtn.setAttribute("aria-label", "Read aloud");
      speakBtn.innerHTML = `<svg class="icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 15a11 11 0 01-9 9M23 8a15 15 0 010 8M11 5a3 3 0 000 6h8V5h-8z"/>
      </svg>`;
      speakBtn.addEventListener("click", () => readAloud(msg.text));

      const regenerateBtn = document.createElement("button");
      regenerateBtn.className = "message-action-btn";
      regenerateBtn.setAttribute("aria-label", "Regenerate response");
      regenerateBtn.innerHTML = `<svg class="icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36M20.49 15a9 9 0 01-14.85 3.36"/>
      </svg>`;
      regenerateBtn.addEventListener("click", () => regenerateResponse(index));

      actionsEl.appendChild(copyBtn);
      actionsEl.appendChild(speakBtn);
      actionsEl.appendChild(regenerateBtn);

      contentEl.appendChild(timestampEl);
      contentEl.appendChild(actionsEl);
    }

    messageEl.appendChild(contentEl);
    app.messagesContainer.appendChild(messageEl);
  });
}

function showEmptyState() {
  app.emptyState.style.display = "flex";
  app.messagesContainer.innerHTML = "";
  app.suggestedPromptsContainer.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      app.promptInput.value = btn.textContent;
      app.promptInput.style.height = "auto";
      app.promptInput.style.height = `${Math.min(app.promptInput.scrollHeight, 200)}px`;
      sendMessage();
    });
  });
}

// ========== MESSAGE ACTIONS ==========
function copyToClipboard(text, button) {
  navigator.clipboard.writeText(text).then(() => {
    button.classList.add("copied");
    setTimeout(() => button.classList.remove("copied"), 2000);
  });
}

function readAloud(text) {
  if (!("speechSynthesis" in window)) {
    alert("Text-to-speech not supported in your browser");
    return;
  }

  // Cancel any ongoing speech
  speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.volume = 1;

  speechSynthesis.speak(utterance);
}

function regenerateResponse(messageIndex) {
  const chat = state.chats[state.currentChatId];
  if (!chat) return;

  // Find the user message before this bot message
  let userMessageText = null;
  for (let i = messageIndex - 1; i >= 0; i--) {
    if (chat.messages[i].role === "user") {
      userMessageText = chat.messages[i].text;
      break;
    }
  }

  if (userMessageText) {
    // Remove the bot message
    chat.messages.splice(messageIndex, 1);
    saveChats();
    renderMessages();
    // Send the message again
    sendMessage(userMessageText);
  }
}

// ========== MESSAGE SENDING ==========
async function sendMessage(overrideText = null) {
  const prompt = (overrideText || app.promptInput.value).trim();
  if (!prompt || state.busy) return;

  if (!state.currentChatId) {
    state.currentChatId = createNewChat();
    updateChatHistoryUI();
  }

  // Hide empty state
  app.emptyState.style.display = "none";

  state.busy = true;
  app.sendBtn.disabled = true;
  app.promptInput.value = "";
  autosize();

  const chat = state.chats[state.currentChatId];

  // Add user message
  chat.messages.push({
    role: "user",
    text: prompt,
    timestamp: new Date().toISOString(),
  });

  // Add thinking indicator
  chat.messages.push({
    role: "bot",
    text: "Thinking...",
    thinking: true,
    timestamp: new Date().toISOString(),
  });

  saveChats();
  renderMessages();
  app.messagesContainer.scrollTop = app.messagesContainer.scrollHeight;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_tokens: 200,
        show_matches: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Remove thinking message and add actual response
    chat.messages.pop();
    chat.messages.push({
      role: "bot",
      text: data.response,
      timestamp: new Date().toISOString(),
    });

    // Update chat title from first user message
    if (chat.messages.length === 2) {
      chat.title = prompt.substring(0, 50);
    }

    saveChats();
    updateChatHistoryUI();
    renderMessages();
    app.messagesContainer.scrollTop = app.messagesContainer.scrollHeight;
  } catch (error) {
    chat.messages.pop();
    chat.messages.push({
      role: "bot",
      text: `Error: ${error.message}`,
      timestamp: new Date().toISOString(),
    });
    saveChats();
    renderMessages();
    app.messagesContainer.scrollTop = app.messagesContainer.scrollHeight;
  } finally {
    state.busy = false;
    app.sendBtn.disabled = false;
    app.promptInput.focus();
  }
}

// ========== INPUT HANDLING ==========
function autosize() {
  app.promptInput.style.height = "auto";
  app.promptInput.style.height = `${Math.min(app.promptInput.scrollHeight, 200)}px`;
}

app.promptInput.addEventListener("input", autosize);

app.promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

app.sendBtn.addEventListener("click", () => sendMessage());

// ========== SIDEBAR CONTROLS ==========
app.sidebarOpenBtn.addEventListener("click", () => {
  app.sidebar.classList.add("open");
});

app.sidebarCloseBtn.addEventListener("click", () => {
  app.sidebar.classList.remove("open");
});

app.newChatBtn.addEventListener("click", () => {
  const chatId = createNewChat();
  selectChat(chatId);
  app.sidebar.classList.remove("open");
  showEmptyState();
});

app.clearHistoryBtn.addEventListener("click", () => {
  if (confirm("Clear all chat history? This cannot be undone.")) {
    state.chats = {};
    state.currentChatId = null;
    saveChats();
    updateChatHistoryUI();
    app.messagesContainer.innerHTML = "";
    showEmptyState();
  }
});

// Close sidebar on message click (mobile)
document.addEventListener("click", (e) => {
  if (window.innerWidth <= 768) {
    if (
      !app.sidebar.contains(e.target) &&
      e.target !== app.sidebarOpenBtn &&
      !e.target.closest(".sidebar-toggle-open")
    ) {
      app.sidebar.classList.remove("open");
    }
  }
});

// ========== INITIALIZATION ==========
function init() {
  state.chats = loadChats();

  if (Object.keys(state.chats).length > 0) {
    const lastChat = Object.values(state.chats).sort(
      (a, b) => new Date(b.createdAt) - new Date(a.createdAt),
    )[0];
    selectChat(lastChat.id);
  } else {
    showEmptyState();
  }

  updateChatHistoryUI();
}

init();
