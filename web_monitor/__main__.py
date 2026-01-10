"""Run the web monitor application."""
from web_monitor.app import app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment monitor web interface")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"Starting experiment monitor on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
