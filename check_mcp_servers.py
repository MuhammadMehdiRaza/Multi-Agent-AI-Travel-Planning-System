"""
check_mcp_servers.py - Connectivity check for the three MCP servers.

Run this before starting the app. It reports which servers were registered, which
were skipped and why, and the full list of tools discovered across all of them.

    python check_mcp_servers.py
"""

import asyncio

from mcp_client import configured_servers, get_tools, missing_servers


async def main() -> None:
    configured = configured_servers()
    skipped = missing_servers()

    print("")
    print("Configured MCP servers")
    print("-" * 70)
    for name in configured:
        print("  ready    " + name)
    for name, reason in sorted(skipped.items()):
        print("  skipped  " + name + "  (" + reason + ")")

    if not configured:
        print("")
        print("No MCP servers are configured. Check your .env file.")
        return

    print("")
    print("Discovering tools...")
    try:
        tools = await get_tools()
    except Exception as exc:
        print("")
        print("Tool discovery failed: " + type(exc).__name__ + ": " + str(exc))
        for index, sub in enumerate(getattr(exc, "exceptions", []), start=1):
            print("  sub-exception " + str(index) + ": " + type(sub).__name__ + ": " + str(sub))
        return

    print("")
    print("Discovered " + str(len(tools)) + " tools")
    print("-" * 70)
    for tool in sorted(tools, key=lambda candidate: candidate.name):
        description = (tool.description or "").strip().splitlines()
        first_line = description[0][:56] if description else ""
        print("  " + tool.name.ljust(46) + first_line)
    print("")


if __name__ == "__main__":
    asyncio.run(main())
