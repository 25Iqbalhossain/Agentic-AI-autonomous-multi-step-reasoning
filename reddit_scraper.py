# reddit_scraper.py
from __future__ import annotations

import os
import asyncio
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# MCP / LangChain
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------
# Env & logging
# ---------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger("agentic-ai-journalist.reddit")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MCP_COMMAND = "npx"
MCP_ARGS = ["@brightdata/mcp"]  # requires Node + package available to npx
MCP_ENV = {
    "BRIGHTDATA_API_TOKEN": os.getenv("BRIGHTDATA_API_TOKEN", ""),
    "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", ""),
}

# Rate-limit MCP tool usage: 1 request / 15s (plus small sleep per topic)
mcp_limiter = AsyncLimiter(1, 15)

# LLM (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Time window (last 14 days)
from datetime import datetime, timedelta
two_weeks_ago = datetime.today() - timedelta(days=14)
two_weeks_ago_str = two_weeks_ago.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------
class MCPOverloadedError(Exception):
    """Raised when the MCP server/tool signals overload/rate limiting."""


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def _require_env() -> None:
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not MCP_ENV["BRIGHTDATA_API_TOKEN"]:
        missing.append("BRIGHTDATA_API_TOKEN")
    if not MCP_ENV["WEB_UNLOCKER_ZONE"]:
        missing.append("WEB_UNLOCKER_ZONE")
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them in your .env to enable Reddit analysis."
        )


def _build_agent(tools: list[Any]) -> Any:
    """Create the LangGraph ReAct agent wired to Groq + MCP tools."""
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1024,
    )
    return create_react_agent(llm, tools)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=15, max=60),
    retry=retry_if_exception_type(MCPOverloadedError),
    reraise=True,
)
async def _process_topic(agent: Any, topic: str) -> str:
    """Query tools via agent and produce a concise analysis summary for one topic."""
    async with mcp_limiter:
        messages = [
            SystemMessage(
                content=(
                    "You are a Reddit analysis expert. Use available tools only. "
                    f"Strictly consider posts after {two_weeks_ago_str} (ignore older items). "
                    "Return a compact, factual synthesis."
                )
            ),
            HumanMessage(
                content=(
                    f"Analyze Reddit discussions about '{topic}'. "
                    "Output:\n"
                    "- Main discussion points\n"
                    "- Key opinions (quote short snippets without usernames)\n"
                    "- Notable trends/patterns\n"
                    "- Overall narrative and sentiment (positive/neutral/negative)\n"
                    "Keep it concise but information-dense."
                )
            ),
        ]

        try:
            resp = await agent.ainvoke({"messages": messages})
            # LangGraph returns a state; final message is last
            last = resp["messages"][-1]
            return last.content if hasattr(last, "content") else str(last)
        except Exception as e:
            # Retry only on overload hints; otherwise bubble up
            text = str(e)
            if "Overloaded" in text or "rate limit" in text.lower():
                raise MCPOverloadedError(text)
            raise


# ---------------------------------------------------------------------
# Public API (used by backend)
# ---------------------------------------------------------------------
async def scrape_reddit_topics(topics: List[str]) -> Dict[str, str]:
    """
    Returns {topic: summary}. Fails open per topic:
    - If MCP/agent fails for a topic, that topic maps to an empty string.
    - If MCP cannot start at all, returns {} (backend can proceed without Reddit).
    """
    # Validate env once (surface early if totally misconfigured)
    try:
        _require_env()
    except Exception as e:
        logger.error("Reddit analysis disabled: %s", e)
        return {}

    server_params = StdioServerParameters(
        command=MCP_COMMAND,
        args=MCP_ARGS,
        env=MCP_ENV,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = _build_agent(tools)

                results: Dict[str, str] = {}

                # Process topics sequentially with rate-limit; you can change to gather()
                for topic in topics:
                    try:
                        summary = await _process_topic(agent, topic)
                        results[topic] = summary
                    except Exception as e:
                        logger.error("Reddit processing failed for '%s': %r", topic, e)
                        results[topic] = ""  # fail-open for this topic
                    # small spacing between topics in addition to limiter
                    await asyncio.sleep(5)

                return results

    except Exception as e:
        # If MCP server cannot start or tools can't load, fail open
        logger.error("Reddit MCP bootstrap failed: %r", e)
        return {}
