#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import webbrowser

from mini_token_chat import ConversationModel, generate_text, load_model

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Identical Twin Chat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/styles.css" />
</head>
<body>
  <div class="ambient ambient-a"></div>
  <div class="ambient ambient-b"></div>
  <main class="app-shell">
    <section class="hero-card">
      <div class="hero-copy">
        <div class="eyebrow">Token retrieval chatbot</div>
        <h1>BetterThanJackGPT</h1>
        <p>
          A compact dark-mode interface for the same paired prompt/response chatbot.
          Type a message, inspect the closest matches, and keep the conversation moving.
        </p>
      </div>
      <div class="hero-stats" id="stats">
        <div>
          <span>Prompts</span>
          <strong id="prompt-count">--</strong>
        </div>
        <div>
          <span>Responses</span>
          <strong id="response-count">--</strong>
        </div>
        <div>
          <span>Vocabulary</span>
          <strong id="vocab-count">--</strong>
        </div>
      </div>
    </section>

    <section class="workspace">
      <div class="chat-panel">
        <div class="chat-toolbar">
          <div class="status-pill"><span class="dot"></span>online</div>
          <label class="toggle">
            <input id="show-matches" type="checkbox" />
            <span>show matches</span>
          </label>
        </div>

        <div id="messages" class="messages" aria-live="polite"></div>

        <div class="composer">
          <textarea id="prompt" rows="1" placeholder="Ask something like: how do I start?" autocomplete="off" spellcheck="false"></textarea>
          <button id="send" type="button">Send</button>
        </div>
        <div class="hint-row">
          <span>Enter to send</span>
          <span>Shift+Enter for a new line</span>
        </div>
      </div>

      <aside class="side-panel">
        <div class="side-card">
          <h2>Why this reply</h2>
          <p>The server returns the best matching prompts when you enable match display.</p>
          <div id="match-list" class="match-list empty">
            <span>No match data yet.</span>
          </div>
        </div>

        <div class="side-card compact">
          <h2>Shortcuts</h2>
          <ul>
            <li>Use the toggle to show the top prompt matches.</li>
            <li>The input grows as you type.</li>
            <li>The layout stays usable on smaller screens.</li>
          </ul>
        </div>
      </aside>
    </section>
  </main>

  <script src="/app.js"></script>
</body>
</html>
"""

STYLES_CSS = """
:root {
  color-scheme: dark;
  --bg: #050816;
  --bg-2: #0a1224;
  --panel: rgba(13, 20, 38, 0.82);
  --panel-strong: rgba(17, 26, 46, 0.96);
  --border: rgba(148, 163, 184, 0.18);
  --text: #e5eefb;
  --muted: #94a3b8;
  --accent: #5eead4;
  --accent-2: #60a5fa;
  --accent-3: #f59e0b;
  --shadow: 0 24px 80px rgba(2, 6, 23, 0.55);
}

* {
  box-sizing: border-box;
}

html, body {
  margin: 0;
  min-height: 100%;
  background:
    radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 32%),
    radial-gradient(circle at 85% 15%, rgba(94, 234, 212, 0.16), transparent 28%),
    linear-gradient(180deg, var(--bg), var(--bg-2));
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body {
  position: relative;
  overflow-x: hidden;
}

.ambient {
  position: fixed;
  inset: auto;
  border-radius: 999px;
  filter: blur(48px);
  pointer-events: none;
  opacity: 0.55;
  z-index: 0;
}

.ambient-a {
  width: 22rem;
  height: 22rem;
  background: rgba(96, 165, 250, 0.16);
  top: -6rem;
  left: -8rem;
}

.ambient-b {
  width: 26rem;
  height: 26rem;
  background: rgba(94, 234, 212, 0.12);
  bottom: -8rem;
  right: -10rem;
}

.app-shell {
  position: relative;
  z-index: 1;
  width: min(1200px, calc(100vw - 2rem));
  margin: 0 auto;
  padding: 1.2rem 0 1.6rem;
}

.hero-card,
.chat-panel,
.side-card {
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

.hero-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  padding: 1.5rem;
  border-radius: 24px;
  animation: lift-in 500ms ease-out both;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.hero-copy h1 {
  margin: 0;
  font-size: clamp(2rem, 4vw, 3.8rem);
  line-height: 0.96;
  max-width: 12ch;
}

.hero-copy p {
  max-width: 60ch;
  margin: 1rem 0 0;
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.65;
}

.hero-stats {
  display: grid;
  gap: 0.8rem;
  grid-template-columns: repeat(3, minmax(92px, 1fr));
  align-self: end;
}

.hero-stats div {
  padding: 0.9rem 1rem;
  border-radius: 18px;
  background: rgba(9, 14, 28, 0.82);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.hero-stats span {
  display: block;
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.15em;
}

.hero-stats strong {
  display: block;
  margin-top: 0.3rem;
  font-size: 1.55rem;
  font-weight: 800;
}

.workspace {
  margin-top: 1rem;
  display: grid;
  grid-template-columns: minmax(0, 1.65fr) minmax(280px, 0.85fr);
  gap: 1rem;
}

.chat-panel,
.side-card {
  border-radius: 24px;
}

.chat-panel {
  display: flex;
  flex-direction: column;
  min-height: 72vh;
  padding: 1rem;
}

.chat-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.35rem 0.2rem 0.9rem;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.75rem;
  border-radius: 999px;
  background: rgba(10, 18, 36, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.14);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 0.72rem;
}

.status-pill .dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 0 0 6px rgba(94, 234, 212, 0.12);
}

.toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  color: var(--muted);
  font-size: 0.9rem;
  user-select: none;
}

.toggle input {
  width: 1.05rem;
  height: 1.05rem;
  accent-color: var(--accent);
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 0.3rem;
  scroll-behavior: smooth;
}

.message {
  display: grid;
  gap: 0.3rem;
  margin: 0.8rem 0;
  animation: rise 220ms ease-out both;
}

.message .meta {
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.bubble {
  width: fit-content;
  max-width: min(72ch, 88%);
  padding: 0.9rem 1rem;
  border-radius: 18px;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}

.user {
  justify-items: end;
}

.user .bubble {
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.26), rgba(59, 130, 246, 0.2));
  border: 1px solid rgba(96, 165, 250, 0.26);
  border-bottom-right-radius: 6px;
}

.bot .bubble {
  background: rgba(13, 20, 38, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-bottom-left-radius: 6px;
}

.bubble.thinking {
  color: var(--muted);
  font-style: italic;
}

.composer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.8rem;
  align-items: end;
  padding-top: 1rem;
}

.composer textarea {
  resize: none;
  min-height: 3.4rem;
  max-height: 10rem;
  padding: 0.95rem 1rem;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(8, 14, 28, 0.95);
  color: var(--text);
  font: inherit;
  line-height: 1.5;
  outline: none;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.composer textarea:focus {
  border-color: rgba(94, 234, 212, 0.55);
  box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.12);
}

.composer button {
  border: 0;
  border-radius: 18px;
  padding: 0.95rem 1.3rem;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: #04111f;
  font-weight: 800;
  cursor: pointer;
  box-shadow: 0 12px 30px rgba(37, 99, 235, 0.24);
  transition: transform 160ms ease, filter 160ms ease;
}

.composer button:hover {
  transform: translateY(-1px);
  filter: brightness(1.05);
}

.composer button:disabled {
  opacity: 0.65;
  cursor: progress;
  transform: none;
}

.hint-row {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 0.65rem;
  color: var(--muted);
  font-size: 0.8rem;
}

.side-panel {
  display: grid;
  gap: 1rem;
}

.side-card {
  padding: 1.1rem;
}

.side-card h2 {
  margin: 0;
  font-size: 1rem;
}

.side-card p,
.side-card li {
  color: var(--muted);
  line-height: 1.6;
}

.side-card ul {
  margin: 0.8rem 0 0;
  padding-left: 1rem;
}

.match-list {
  display: grid;
  gap: 0.65rem;
  margin-top: 1rem;
}

.match-card {
  padding: 0.85rem 0.9rem;
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: rgba(10, 18, 36, 0.8);
}

.match-card .score {
  color: var(--accent);
  font-weight: 700;
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.match-card .prompt {
  margin-top: 0.35rem;
  font-size: 0.95rem;
}

.match-card .response {
  margin-top: 0.5rem;
  color: var(--muted);
  font-size: 0.9rem;
}

.match-list.empty {
  min-height: 140px;
  align-items: center;
  justify-items: center;
  border: 1px dashed rgba(148, 163, 184, 0.18);
  border-radius: 18px;
  color: var(--muted);
}

.compact {
  opacity: 0.98;
}

@keyframes lift-in {
  from {
    transform: translateY(12px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

@keyframes rise {
  from {
    transform: translateY(8px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

@media (max-width: 960px) {
  .hero-card,
  .workspace {
    grid-template-columns: 1fr;
  }

  .hero-stats {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .app-shell {
    width: min(100vw - 1rem, 1200px);
    padding-top: 0.5rem;
  }

  .hero-card,
  .chat-panel,
  .side-card {
    border-radius: 20px;
  }

  .hero-card {
    padding: 1.2rem;
  }

  .hero-stats {
    grid-template-columns: 1fr;
  }

  .chat-panel {
    min-height: 68vh;
  }

  .composer {
    grid-template-columns: 1fr;
  }

  .hint-row {
    flex-direction: column;
    gap: 0.25rem;
  }

  .bubble {
    max-width: 100%;
  }
}
"""

APP_JS = """
const messages = document.getElementById('messages');
const promptInput = document.getElementById('prompt');
const sendButton = document.getElementById('send');
const showMatches = document.getElementById('show-matches');
const matchList = document.getElementById('match-list');
const promptCount = document.getElementById('prompt-count');
const responseCount = document.getElementById('response-count');
const vocabCount = document.getElementById('vocab-count');

const state = {
  busy: false,
  history: [],
};

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function createMessage(role, text, meta) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const metaNode = document.createElement('div');
  metaNode.className = 'meta';
  metaNode.textContent = meta;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  wrap.append(metaNode, bubble);
  return { wrap, bubble };
}

function renderMatchList(matches) {
  matchList.innerHTML = '';
  if (!matches || !matches.length) {
    matchList.className = 'match-list empty';
    const empty = document.createElement('span');
    empty.textContent = 'No match data yet.';
    matchList.appendChild(empty);
    return;
  }

  matchList.className = 'match-list';
  matches.forEach((match, index) => {
    const card = document.createElement('div');
    card.className = 'match-card';

    const score = document.createElement('div');
    score.className = 'score';
    score.textContent = `match ${index + 1} · ${match.score.toFixed(3)}`;

    const prompt = document.createElement('div');
    prompt.className = 'prompt';
    prompt.textContent = match.prompt;

    const response = document.createElement('div');
    response.className = 'response';
    response.textContent = match.response;

    card.append(score, prompt, response);
    matchList.appendChild(card);
  });
}

function autosize() {
  promptInput.style.height = 'auto';
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

async function sendMessage() {
  const prompt = promptInput.value.trim();
  if (!prompt || state.busy) {
    return;
  }

  state.busy = true;
  sendButton.disabled = true;
  promptInput.value = '';
  autosize();

  const userMessage = createMessage('user', prompt, 'you');
  messages.appendChild(userMessage.wrap);
  scrollToBottom();

  const thinking = createMessage('bot', 'thinking...', 'bot');
  thinking.bubble.classList.add('thinking');
  messages.appendChild(thinking.wrap);
  scrollToBottom();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: 40,
        show_matches: showMatches.checked,
      }),
    });

    if (!response.ok) {
      throw new Error(`request failed with ${response.status}`);
    }

    const payload = await response.json();
    thinking.wrap.remove();

    const botMessage = createMessage('bot', payload.response, 'bot');
    messages.appendChild(botMessage.wrap);
    renderMatchList(payload.matches);
    scrollToBottom();

    state.history.push({ user: prompt, bot: payload.response });
  } catch (error) {
    thinking.wrap.remove();
    const failure = createMessage('bot', `Sorry, I could not reach the model. ${error.message}`, 'bot');
    messages.appendChild(failure.wrap);
    scrollToBottom();
  } finally {
    state.busy = false;
    sendButton.disabled = false;
    promptInput.focus();
  }
}

promptInput.addEventListener('input', autosize);
promptInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

sendButton.addEventListener('click', sendMessage);
showMatches.addEventListener('change', () => {
  if (showMatches.checked && state.history.length) {
    const last = state.history[state.history.length - 1];
    renderMatchList([{ score: 1.0, prompt: last.user, response: last.bot }]);
  } else if (!showMatches.checked) {
    renderMatchList([]);
  }
});

async function bootstrap() {
  const res = await fetch('/api/meta');
  const meta = await res.json();
  promptCount.textContent = meta.prompt_count;
  responseCount.textContent = meta.response_count;
  vocabCount.textContent = meta.vocab_size;

  const welcome = createMessage('bot', 'Ask me something and I will score the closest prompts.', 'bot');
  messages.appendChild(welcome.wrap);
  scrollToBottom();
  promptInput.focus();
}

bootstrap().catch(() => {
  const fallback = createMessage('bot', 'The server is up, but the metadata endpoint did not respond.', 'bot');
  messages.appendChild(fallback.wrap);
});
"""


class ChatServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], model: ConversationModel):
        super().__init__(server_address, handler_class)
        self.model = model


class ChatRequestHandler(BaseHTTPRequestHandler):
    server_version = "IdenticalTwinChat/1.0"

    def _send_text(self, content: str, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_text(json.dumps(payload, ensure_ascii=True), content_type="application/json; charset=utf-8", status=status)

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def model(self) -> ConversationModel:
        return self.server.model  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_text(INDEX_HTML)
        elif path == "/styles.css":
            self._send_text(STYLES_CSS, content_type="text/css; charset=utf-8")
        elif path == "/app.js":
            self._send_text(APP_JS, content_type="application/javascript; charset=utf-8")
        elif path == "/api/meta":
            self._send_json(
                {
                    "prompt_count": len(self.model.prompts),
                    "response_count": len(self.model.responses),
                    "vocab_size": len(self.model.vocab),
                }
            )
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except Exception as exc:
            self._send_json({"error": f"invalid request body: {exc}"}, status=400)
            return

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send_json({"error": "prompt is required"}, status=400)
            return

        max_tokens = int(payload.get("max_tokens", 40))
        show_matches = bool(payload.get("show_matches", False))
        response = generate_text(self.model, prompt, max_tokens, show_matches=show_matches, debug=False)
        matches = self.model.find_matches(prompt, top_n=5)

        self._send_json(
            {
                "prompt": prompt,
                "response": response,
                "matches": [
                    {
                        "score": hit.score,
                        "prompt": hit.prompt,
                        "response": hit.response,
                    }
                    for hit in matches
                ],
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser UI for the identical twin chatbot")
    parser.add_argument("--model", type=Path, default=Path("model.json"), help="Path to the saved chatbot model")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument("--open", action="store_true", help="Open the browser automatically")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = load_model(args.model)
    server = ChatServer((args.host, args.port), ChatRequestHandler, model)
    url = f"http://{args.host}:{args.port}/"
    print(f"serving on {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
