"""HTTP server for SPLM Chat"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from splm import ConversationModel, generate_text


class ChatServer(ThreadingHTTPServer):
    """HTTP server with access to the conversation model"""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        model: ConversationModel,
    ):
        super().__init__(server_address, handler_class)
        self.model = model


class ChatRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for SPLM Chat API and web assets"""

    server_version = "SPLM Chat/1.0"

    def _send_text(
        self,
        content: str,
        content_type: str = "text/html; charset=utf-8",
        status: int = 200,
    ) -> None:
        """Send a text response with proper headers"""
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, file_path: Path, content_type: str) -> None:
        """Send a file response"""
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._send_json({"error": "file not found"}, status=404)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        """Send a JSON response"""
        self._send_text(
            json.dumps(payload, ensure_ascii=True),
            content_type="application/json; charset=utf-8",
            status=status,
        )

    def log_message(self, format_spec: str, *args: Any) -> None:
        """Suppress default logging"""
        _ = (format_spec, args)

    @property
    def model(self) -> ConversationModel:
        """Get the conversation model from the server"""
        return self.server.model  # type: ignore[attr-defined]

    @property
    def web_dir(self) -> Path:
        """Get the web directory path"""
        return Path(__file__).parent / "web"

    # pylint: disable=invalid-name
    def do_GET(self) -> None:
        """Handle GET requests"""
        path = urlparse(self.path).path

        if path == "/":
            # Serve index.html
            html_file = self.web_dir / "index.html"
            self._send_file(html_file, "text/html; charset=utf-8")

        elif path == "/styles.css":
            # Serve styles.css
            css_file = self.web_dir / "styles.css"
            self._send_file(css_file, "text/css; charset=utf-8")

        elif path == "/app.js":
            # Serve app.js
            js_file = self.web_dir / "app.js"
            self._send_file(js_file, "application/javascript; charset=utf-8")

        elif path == "/api/meta":
            # Return metadata
            self._send_json(
                {
                    "prompt_count": len(self.model.prompts),
                    "response_count": len(self.model.responses),
                    "vocab_size": len(self.model.vocab),
                }
            )

        else:
            self._send_json({"error": "not found"}, status=404)

    # pylint: disable=invalid-name
    def do_POST(self) -> None:
        """Handle POST requests"""
        path = urlparse(self.path).path

        if path != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json(
                {"error": f"invalid request body: {exc}"}, status=400
            )
            return

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send_json({"error": "prompt is required"}, status=400)
            return

        max_tokens = int(payload.get("max_tokens", 40))
        show_matches = bool(payload.get("show_matches", False))

        response = generate_text(
            self.model,
            prompt,
            max_tokens,
            show_matches=show_matches,
            debug=False,
        )
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