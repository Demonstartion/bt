import os
import logging
import aiohttp

class SearchService:
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serp_api_key = os.getenv("SERP_API_KEY")
        
    async def search(self, query: str, max_results: int = 5) -> str:
        try:
            if self.tavily_api_key:
                result = await self._search_tavily(query, max_results)
                if result:
                    return result
            if self.serp_api_key:
                result = await self._search_serpapi(query, max_results)
                if result:
                    return result
            logging.warning("No search API keys configured")
            return ""
        except Exception as e:
            logging.error(f"Error in search service: {e}")
            return ""
    
    async def _search_tavily(self, query: str, max_results: int):
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_tavily_results(data)
                    else:
                        logging.error(f"Tavily API error: {response.status}")
                        return None
        except Exception as e:
            logging.error(f"Tavily search error: {e}")
            return None
    
    async def _search_serpapi(self, query: str, max_results: int):
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.serp_api_key,
                "engine": "google",
                "num": max_results
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_serpapi_results(data)
                    else:
                        logging.error(f"SerpAPI error: {response.status}")
                        return None
        except Exception as e:
            logging.error(f"SerpAPI search error: {e}")
            return None
    
    def _format_tavily_results(self, data: dict) -> str:
        try:
            results = []
            if "answer" in data and data["answer"]:
                results.append(f"Quick Answer: {data['answer']}")
            if "results" in data:
                for item in data["results"][:5]:
                    title = item.get("title", "")
                    content = item.get("content", "")
                    url = item.get("url", "")
                    if title and content:
                        results.append(f"• {title}\n  {content}\n  Source: {url}")
            return "\n\n".join(results) if results else ""
        except Exception as e:
            logging.error(f"Error formatting Tavily results: {e}")
            return ""
    
    def _format_serpapi_results(self, data: dict) -> str:
        try:
            results = []
            if "answer_box" in data:
                answer = data["answer_box"].get("answer", "")
                if answer:
                    results.append(f"Quick Answer: {answer}")
            if "organic_results" in data:
                for item in data["organic_results"][:5]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    if title and snippet:
                        results.append(f"• {title}\n  {snippet}\n  Source: {link}")
            return "\n\n".join(results) if results else ""
        except Exception as e:
            logging.error(f"Error formatting SerpAPI results: {e}")
            return ""