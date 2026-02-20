cat <<EOF > mcp_hello.py
class MCPServer:
    def say_hello(self):
        print("Hello from MCP!")

server = MCPServer()
server.say_hello()
EOF