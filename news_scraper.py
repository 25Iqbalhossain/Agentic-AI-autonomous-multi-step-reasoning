import asyncio
import os
from typing import Dict, List

from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
# ✅ Replace Anthropic with Groq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from utils import (
    generate_news_urls_to_scrape,
    scrape_with_brightdata,
    clean_html_to_text,
    extract_headlines,
    summarize_with_ollama   # (kept this, you can still use Ollama if needed)
)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()


class NewsScraper:
    _rate_limiter = AsyncLimiter(5, 1)  # 5 requests/second

    def __init__(self):
        # ✅ Initialize Groq LLM here
        self.groq_model = ChatGroq(
            model="llama3-70b-8192",  # or "llama3-8b-8192" if you want faster/cheaper
            temperature=0,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY")
        )

    async def _summarize_with_groq(self, headlines: List[str]) -> str:
        """Summarize headlines using Groq LLM instead of Anthropic"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a news summarizer. Analyze the given headlines and "
                    "generate a concise, neutral, and informative summary of the news."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(headlines),
            },
        ]

        try:
            response = await self.groq_model.ainvoke({"messages": messages})
            return response["messages"][-1].content
        except Exception as e:
            return f"Groq summarization error: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def scrape_news(self, topics: List[str]) -> Dict[str, str]:
        """Scrape and analyze news articles"""
        results = {}
        
        for topic in topics:
            async with self._rate_limiter:
                try:
                    urls = generate_news_urls_to_scrape([topic])
                    search_html = scrape_with_brightdata(urls[topic])
                    clean_text = clean_html_to_text(search_html)
                    headlines = extract_headlines(clean_text)

                    # ✅ Now summarized by Groq instead of Anthropic
                    summary = await self._summarize_with_groq(headlines)

                    results[topic] = summary
                except Exception as e:
                    results[topic] = f"Error: {str(e)}"

                await asyncio.sleep(1)  # Avoid overwhelming news sites

        return {"news_analysis": results}
