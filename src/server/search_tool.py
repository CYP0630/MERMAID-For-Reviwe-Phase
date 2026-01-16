from __future__ import annotations
import os
import json
import requests
from typing import Dict, Optional

from fastmcp import FastMCP

mcp = FastMCP(
    name="Google Search Tool",
    instructions="""Google Search MCP Server for fact-checking tasks."""
    )


class SearchAPI:
    """Class for querying the SearchAPI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = "uYFP2VggL6fDF5eXb29VGBcV"
        self.base_url = "https://www.searchapi.io/api/v1/search"
        self.timeout = 10

    def google_search(self, query: str, num_results: int = 5) -> Dict:
        """
        Search Google and return relevant articles

        Args:
            query: Search query string
            num_results: Number of results to return (max 10)

        Returns:
            Dict containing search results and metadata
        """
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": min(num_results, 10),
            }

            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Extract relevant results efficiently
            results = []
            organic_results = data.get('organic_results', [])

            # Organize the retrieved results
            results = [
                {
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'source': result.get('source', ''),
                    'position': result.get('position', 0)
                }
                for result in organic_results[:num_results]
            ]

            # Include answer box (if available, often contains direct answers)
            answer_box = data.get('answer_box', {})
            if answer_box:
                results.insert(0, {
                    'title': 'Direct Answer',
                    'link': answer_box.get('link', ''),
                    'snippet': answer_box.get('answer', answer_box.get('snippet', '')),
                    'source': 'Answer Box',
                    'position': 0
                })

            return {
                'query': query,
                'results': results,
                'total': len(results),
                'search_metadata': {
                    'time': data.get('search_metadata', {}).get('total_time_taken', 0)
                }
            }

        except requests.exceptions.Timeout:
            return {'error': 'Search request timed out', 'results': []}
        except requests.exceptions.RequestException as e:
            return {'error': f'API request failed: {str(e)}', 'results': []}
        except json.JSONDecodeError as e:
            return {'error': 'Invalid response format', 'results': []}
        except Exception as e:
            return {'error': str(e), 'results': []}


search_api = SearchAPI()
@mcp.tool(name="search_google", description="Retrieve Google search results for a given query.")
def search_google(query: str, max_results: int = 10) -> Dict:
    """
    Search from Google related to a query.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Dictionary containing Google search results with titles, links, snippets, and metadata
    """
    return search_api.google_search(query, max_results)

# ------------------------------- Entry Point -------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")