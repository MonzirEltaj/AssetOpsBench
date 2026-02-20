# hello_mcp.py
import time

class HelloMCP:
    """A minimal MCP agent for testing."""
    def __init__(self):
        print("HelloMCP initialized")

    def handle_request(self, request):
        print(f"Received request: {request}")
        return "Hello from MCP!"

if __name__ == "__main__":
    agent = HelloMCP()
    print("MCP agent running. Press Ctrl+C to stop.")
    # simulate a server listening
    while True:
        fake_request = "ping"
        response = agent.handle_request(fake_request)
        print(f"Response: {response}")
        time.sleep(5)