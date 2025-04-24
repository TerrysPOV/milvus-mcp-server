import { createTool, FastMCP } from "fastmcp/server";
import { Milvus } from "fastmcp-milvus";

// Example Milvus tool: create collection
const createCollection = createTool({
  name: "milvus_create_collection",
  description: "Create a collection in Milvus.",
  parameters: {
    name: {
      type: "string",
      description: "Name of the collection to create",
    },
    dimension: {
      type: "number",
      default: 1536,
      description: "Number of dimensions for the vectors",
    },
  },
  run: async ({ name, dimension }, ctx) => {
    const client = await ctx.namespace.use(Milvus);
    await client.createCollection(name, dimension);
    return `Collection '${name}' created with dimension ${dimension}.`;
  },
});

// Create and configure the server
const mcp = new FastMCP({
  tools: [createCollection],
  namespaces: [Milvus],
});

async function main() {
  await mcp.run();
}

main();
