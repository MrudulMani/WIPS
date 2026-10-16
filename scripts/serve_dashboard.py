"""
WIPS Interactive Dashboard Server
Launches a lightweight local HTTP server to view the 2D floor plans,
AP signal coverage, and benchmark comparisons.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)
        
    def end_headers(self):
        # Enable CORS and disable aggressive caching for development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

def serve():
    os.chdir(BASE_DIR)
    url = f"http://localhost:{PORT}/web/index.html"
    print("=" * 60)
    print("🌐 WIPS Interactive Dashboard")
    print("=" * 60)
    print(f"Server running at: {url}")
    print("Press Ctrl+C to terminate.")
    print("=" * 60)
    
    try:
        webbrowser.open(url)
    except Exception:
        pass

    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    serve()
