"""MCP server exposing tweets from an RSS feed."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

import feedparser
import httpx
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field, HttpUrl

from . import RSS_FEED_URL


class FeedItem(BaseModel):
    """Structured representation of an RSS feed entry."""

    id: str = Field(description="Stable identifier for the post")
    title: str = Field(description="Title extracted from the RSS entry")
    content: str = Field(description="Plain text content of the tweet or reply")
    url: HttpUrl = Field(description="Direct link to the post on X.com")
    author: Optional[str] = Field(default=None, description="Author if present in the feed")
    published_at: Optional[datetime] = Field(
        default=None, description="Publication timestamp when available"
    )
    is_reply: bool = Field(default=False, description="Best-effort flag indicating replies")


@dataclass(slots=True)
class ParsedEntry:
    """Internal container for feed data before validation."""

    identifier: str
    title: str
    content: str
    url: str
    author: Optional[str]
    published_at: Optional[datetime]
    is_reply: bool


def _strip_html(value: str) -> str:
    """Remove simple HTML tags from the feed content."""

    if not value:
        return ""

    import html
    import re

    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return " ".join(text.split())


def _parse_entry(entry: Any) -> ParsedEntry:
    link = getattr(entry, "link", "")
    identifier = getattr(entry, "id", None) or getattr(entry, "guid", None) or link or "unknown"
    title = getattr(entry, "title", "").strip()
    summary = getattr(entry, "summary", "")
    content = _strip_html(summary or title)
    published = getattr(entry, "published_parsed", None)
    published_at: Optional[datetime] = None
    if published:
        try:
            published_at = datetime(*published[:6])
        except (ValueError, TypeError):
            published_at = None

    author = getattr(entry, "author", None)

    lowered = (title + " " + summary).lower()
    is_reply = "reply" in lowered or " replying to " in lowered

    return ParsedEntry(
        identifier=str(identifier),
        title=title,
        content=content,
        url=link,
        author=author,
        published_at=published_at,
        is_reply=is_reply,
    )


async def _fetch_entries(limit: int) -> List[FeedItem]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(RSS_FEED_URL)
            response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"Failed to fetch RSS feed: {exc}") from exc

    feed = feedparser.parse(response.content)
    items: List[FeedItem] = []

    for entry in feed.entries[:limit]:
        parsed = _parse_entry(entry)
        try:
            model = FeedItem(
                id=parsed.identifier,
                title=parsed.title or parsed.content,
                content=parsed.content,
                url=parsed.url,
                author=parsed.author,
                published_at=parsed.published_at,
                is_reply=parsed.is_reply,
            )
            items.append(model)
        except Exception:
            continue

    return items


mcp = FastMCP(
    name="Lat3ntG3nius RSS",  # type: ignore[arg-type]
    instructions="Fetches latest tweets and replies from a pre-configured RSS feed.",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8000")),
    streamable_http_path="/mcp",
    json_response=True,
)


@mcp.tool(name="fetch_posts", description="Get recent X.com posts from the RSS feed")
async def fetch_posts(
    limit: int = 10,
    include_replies: bool = True,
    ctx: Context[ServerSession, None] | None = None,
) -> List[FeedItem]:
    """Return the latest posts from the RSS feed."""

    limit = max(1, min(limit, 50))
    if ctx is not None:
        await ctx.info(f"Fetching up to {limit} entries from RSS feed")

    try:
        items = await _fetch_entries(limit)
    except RuntimeError as exc:
        if ctx is not None:
            await ctx.error(str(exc))
        raise
    if not include_replies:
        items = [item for item in items if not item.is_reply]

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RSS MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default=os.environ.get("MCP_TRANSPORT", "streamable-http"),
        help="Transport to use when serving",
    )
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.run("streamable-http")
    elif args.transport == "sse":
        mcp.run("sse")
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
