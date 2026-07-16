# 网页搜索工具
from tavily import TavilyClient
from app.config import settings


class WebSearchError(Exception):
    """网页搜索异常"""
    pass


def search_web(query: str, max_results: int = 5) -> dict:
    """
    使用 Tavily API 搜索互联网内容，帮助用户发现最近值得阅读的书籍或获取书评、推荐等信息。

    Args:
        query: 搜索关键词（必填）
        max_results: 返回结果数量，默认 5，最大 10

    Returns:
        包含搜索结果的字典
    """
    if not isinstance(query, str):
        raise WebSearchError("搜索词格式不正确")
    query = query.strip()
    if not query:
        raise WebSearchError("搜索词不能为空")
    if len(query) > 200:
        raise WebSearchError("搜索词长度不能超过 200 个字符")

    if not isinstance(max_results, int) or max_results < 1 or max_results > 10:
        raise WebSearchError("max_results 必须为 1-10 的整数")

    api_key = settings.tavily_api_key.get_secret_value()
    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(query=query, max_results=max_results)
    except Exception:
        raise WebSearchError("网络搜索请求失败，请稍后重试")

    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise WebSearchError("搜索返回数据格式异常")

    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")[:500],
        })

    return {
        "ok": True,
        "query": query,
        "results": results,
    }