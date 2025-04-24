# Claude-Compatible Milvus MCP Server
This repo provides everything needed to connect Anthropic's Claude (via Claude Desktop) to a Milvus vector database using the MCP (Model Context Protocol). It includes:

- 🧠 A fully MCP-compatible FastMCP server
- 🗃️ Milvus setup (via Docker)
- 🛠️ Shell script to launch the server correctly for Claude
- ✅ Official Milvus tool names for Claude compatibility

## 📦 Folder Structure

.
├── milvus-standalone/                 # Contains Docker Compose for Milvus
│   ├── docker-compose.yml
│   └── config/                        # Milvus etcd config files (if needed)
├── fastmcp/
│   ├── milvus-server/                # Your custom MCP server files
│   │   ├── server.py                 # MCP server code
│   │   └── run_milvus_server.sh     # Shell script to launch server for Claude
│   ├── .venv/                        # uv-managed Python virtual env
│   └── pyproject.toml                # FastMCP + dependencies
├── claude_desktop_config.json        # Add this entry to Claude's config
└── README.md

## 🚀 Setup Guide

1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/milvus-mcp-server
cd milvus-mcp-server

2. Run Milvus (Docker)
Ensure Docker is installed and running.
cd milvus-standalone
docker-compose up -d

3. Install uv (if you haven’t yet)
brew install uv

4. Sync Python Environment
Inside fastmcp/:
cd fastmcp
uv sync

5. Test the Server
uv run python milvus-server/server.py
You should see logs like:

pgsql
Copy
Edit
INFO: Connected to Milvus
✅ Server loaded and tools are registered

6. Update Claude Config
Update claude_desktop_config.json like so:
}
"milvus": {
  "command": "/full/path/to/run_milvus_server.sh",
  "args": []
}

Make sure to restart Claude after this.

7. Ask Claude
Try:

“Call the Milvus tool to list all collections.”

It should now connect and execute.

## 🧠 Tool Naming Rules (Critical!)
Claude only recognizes tools if they exactly match Milvus naming conventions. Do not rename tools. Use:

create_collection

list_collections

search

etc.

Use @mcp.tool(name="create_collection") — the name is case-sensitive.

## 🧪 Debugging Tips
Ensure Milvus Docker is healthy (docker ps)

Check FastMCP server logs in the terminal

Add print statements in tools to debug

Make sure .venv and uv are used consistently

Don't use emojis or non-ASCII characters in tool descriptions

## 📌 Next Steps
 Add more tools (delete_collection, upsert, etc.)

 Add test scripts (test_server.py)

 Add GitHub Actions for validation

## 🧪 Tests
To manually run tool tests, use the test_server_full.py or Claude Desktop interface.

## 🤝 Credits
- Built by @TerrysPOV
- Powered by FastMCP
- Vectors stored in Milvus
- Embeddings via OpenAI (text-embedding-ada-002)

## 📜 License
MIT