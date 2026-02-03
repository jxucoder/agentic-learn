"""Papers extension for searching and reading research papers."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from agentic_learn.core.extension import Extension, ExtensionAPI
from agentic_learn.core.tool import Tool, ToolContext, ToolParameter
from agentic_learn.core.types import ToolResult


class PaperSearchTool(Tool):
    """Search for research papers on arXiv and Semantic Scholar."""

    name = "papers_search"
    description = """Search for research papers on arXiv and Semantic Scholar.

Use this to:
- Find papers on a specific topic
- Look up recent research in an area
- Find papers by author
- Get SOTA methods for a task

Returns: paper titles, authors, abstracts, and links.

Examples:
- query="transformer attention mechanism"
- query="BERT language model" source="arxiv"
- query="Yann LeCun" type="author\""""

    parameters = [
        ToolParameter(
            name="query",
            type=str,
            description="Search query (topic, title keywords, or author name)",
            required=True,
        ),
        ToolParameter(
            name="source",
            type=str,
            description="Source to search: 'arxiv', 'semantic_scholar', or 'all' (default: all)",
            required=False,
            default="all",
        ),
        ToolParameter(
            name="max_results",
            type=int,
            description="Maximum number of results (default: 5)",
            required=False,
            default=5,
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        query: str,
        source: str = "all",
        max_results: int = 5,
    ) -> ToolResult:
        """Search for papers."""
        results = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            if source in ("arxiv", "all"):
                try:
                    arxiv_results = await self._search_arxiv(client, query, max_results)
                    results.extend(arxiv_results)
                except Exception as e:
                    results.append({"error": f"arXiv search failed: {e}"})

            if source in ("semantic_scholar", "all"):
                try:
                    ss_results = await self._search_semantic_scholar(client, query, max_results)
                    results.extend(ss_results)
                except Exception as e:
                    results.append({"error": f"Semantic Scholar search failed: {e}"})

        if not results:
            return ToolResult(
                tool_call_id="",
                content="No papers found for query.",
            )

        # Format results
        formatted = [f"Found {len(results)} papers for '{query}':", "=" * 60, ""]

        for i, paper in enumerate(results[:max_results], 1):
            if "error" in paper:
                formatted.append(f"{i}. Error: {paper['error']}")
                continue

            formatted.append(f"{i}. {paper.get('title', 'Unknown Title')}")
            formatted.append(f"   Authors: {paper.get('authors', 'Unknown')}")
            formatted.append(f"   Source: {paper.get('source', 'Unknown')}")
            if paper.get("year"):
                formatted.append(f"   Year: {paper['year']}")
            if paper.get("url"):
                formatted.append(f"   URL: {paper['url']}")
            if paper.get("abstract"):
                abstract = paper["abstract"][:300] + "..." if len(paper["abstract"]) > 300 else paper["abstract"]
                formatted.append(f"   Abstract: {abstract}")
            formatted.append("")

        return ToolResult(
            tool_call_id="",
            content="\n".join(formatted),
            metadata={"count": len(results)},
        )

    async def _search_arxiv(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search arXiv API."""
        import xml.etree.ElementTree as ET

        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = await client.get(url, params=params)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)
            link = entry.find("atom:id", ns)

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None and name.text:
                    authors.append(name.text)

            results.append({
                "title": title.text.strip() if title is not None and title.text else "Unknown",
                "authors": ", ".join(authors[:5]) + ("..." if len(authors) > 5 else ""),
                "abstract": summary.text.strip() if summary is not None and summary.text else "",
                "year": published.text[:4] if published is not None and published.text else None,
                "url": link.text if link is not None else None,
                "source": "arXiv",
            })

        return results

    async def _search_semantic_scholar(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search Semantic Scholar API."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,year,url",
        }

        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        results = []

        for paper in data.get("data", []):
            authors = paper.get("authors", [])
            author_names = [a.get("name", "") for a in authors[:5]]

            results.append({
                "title": paper.get("title", "Unknown"),
                "authors": ", ".join(author_names) + ("..." if len(authors) > 5 else ""),
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "url": paper.get("url"),
                "source": "Semantic Scholar",
            })

        return results


class PaperReadTool(Tool):
    """Read and summarize a paper from arXiv."""

    name = "papers_read"
    description = """Fetch and read a paper from arXiv.

Provide an arXiv ID (e.g., '2103.14030') or URL to get:
- Full abstract
- Paper metadata
- Key sections (if available)

Note: Full PDF parsing requires additional setup."""

    parameters = [
        ToolParameter(
            name="paper_id",
            type=str,
            description="arXiv paper ID (e.g., '2103.14030') or full URL",
            required=True,
        ),
    ]

    async def execute(
        self,
        ctx: ToolContext,
        paper_id: str,
    ) -> ToolResult:
        """Read paper details."""
        import xml.etree.ElementTree as ET

        # Extract ID from URL if needed
        if "arxiv.org" in paper_id:
            paper_id = paper_id.split("/")[-1].replace(".pdf", "")

        url = f"http://export.arxiv.org/api/query?id_list={paper_id}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", ns)
        if entry is None:
            return ToolResult(
                tool_call_id="",
                content=f"Paper not found: {paper_id}",
                is_error=True,
            )

        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        updated = entry.find("atom:updated", ns)

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None and name.text:
                authors.append(name.text)

        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term")
            if term:
                categories.append(term)

        formatted = [
            f"Paper: {paper_id}",
            "=" * 60,
            "",
            f"Title: {title.text.strip() if title is not None and title.text else 'Unknown'}",
            "",
            f"Authors: {', '.join(authors)}",
            "",
            f"Published: {published.text[:10] if published is not None and published.text else 'Unknown'}",
            f"Updated: {updated.text[:10] if updated is not None and updated.text else 'Unknown'}",
            "",
            f"Categories: {', '.join(categories)}",
            "",
            "Abstract:",
            "-" * 40,
            summary.text.strip() if summary is not None and summary.text else "No abstract available",
            "",
            f"PDF: https://arxiv.org/pdf/{paper_id}.pdf",
            f"HTML: https://arxiv.org/abs/{paper_id}",
        ]

        return ToolResult(
            tool_call_id="",
            content="\n".join(formatted),
        )


class PapersExtension(Extension):
    """Extension for searching and reading research papers."""

    name = "papers"
    description = "Search and read research papers from arXiv and Semantic Scholar"
    version = "0.1.0"

    def setup(self, api: ExtensionAPI) -> None:
        """Register paper tools."""
        api.register_tool(PaperSearchTool())
        api.register_tool(PaperReadTool())

        # Register commands
        api.register_command(
            "papers",
            "Search for papers: /papers <query>",
            self._papers_command,
        )

    async def _papers_command(self, ctx: Any, args: list[str]) -> None:
        """Handle /papers command."""
        if not args:
            print("Usage: /papers <search query>")
            return

        query = " ".join(args)
        tool = PaperSearchTool()
        result = await tool.execute(ctx, query=query)
        print(result.content)
