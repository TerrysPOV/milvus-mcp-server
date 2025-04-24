import { FastMCP } from "./fastmcp/src/FastMCP";
import { Milvus } from "./Milvus";
import type { Tool } from "./fastmcp/src/types";


const createCollection: Tool<{ name: string; dimension: number }, string> = {
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
    async run({ name, dimension }, ctx) {
      const client = await ctx.namespace.use(Milvus);
      await client.createCollection(name, dimension);
      return `Collection '${name}' created with dimension ${dimension}.`;
    },
  };
    

// Set up and run the FastMCP server
const mcp = new FastMCP({
  tools: [createCollection],
  namespaces: [Milvus],
});

async function main() {
  await mcp.run(); // now available since we patched FastMCP
}

main();
