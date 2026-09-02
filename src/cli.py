"""CLI and entry point for SPLM Chat"""

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from splm import load_model

from .server import ChatServer, ChatRequestHandler


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for SPLM Chat"""
    parser = argparse.ArgumentParser(
        prog="splm-chat",
        description="SPLM Chat - Modern AI Chat Interface",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.json",
        help="Path to the model JSON file (default: model.json)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the app in the default browser",
    )
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Run the SPLM Chat server"""
    parser = build_parser()
    parsed = parser.parse_args(args)

    # Load the model
    model_path = Path(parsed.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    model = load_model(model_path)

    # Create and run server
    server_address = (parsed.host, parsed.port)
    server = ChatServer(server_address, ChatRequestHandler, model)

    try:
        url = f"http://{parsed.host}:{parsed.port}/"
        print(f"✓ serving on {url}")

        if parsed.open:
            webbrowser.open(url)

        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ shutting down...")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
