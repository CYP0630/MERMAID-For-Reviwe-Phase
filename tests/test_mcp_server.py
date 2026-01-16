import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Dict, Any, List


scripts = [
    "src/server/search_tool.py",
    "src/server/wikipedia_tool.py"
]

async def connect_to_servers(scripts: List[str]):
        """
        Connect to MCP tool servers specified by script paths.

        Args:
            scripts: List of paths to tool server scripts

        Raises:
            RuntimeError: If duplicate tool names are found
        """
        sessions: Dict[str, ClientSession] = {}

        async with AsyncExitStack() as exit_stack:
            for script in scripts:
                path = Path(script)
                # Determine command based on file extension
                cmd = "python" if path.suffix == ".py" else "node"
                params = StdioServerParameters(command=cmd, args=[str(path)])

                # Create stdio client and session
                stdio, write = await exit_stack.enter_async_context(stdio_client(params))
                session = await exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()

                # Register tools from this session
                for tool in (await session.list_tools()).tools:
                    if tool.name in sessions:
                        raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                    sessions[tool.name] = session

            print("Connected tools:", list(sessions.keys()))


asyncio.run(connect_to_servers(scripts))