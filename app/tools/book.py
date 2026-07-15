# 调用go图书接口
import httpx
from app.config import settings

class BookAPIError(Exception):
    """图书 API 调用异常"""
    pass

def search_books(keyword: str, page: int = 1, page_size: int = 5) -> dict:
    """
    搜索图书

    Args:
        keyword: 搜索关键词（必填）
        page: 页码，默认 1
        page_size: 每页数量，默认 5

    Returns:
        包含图书列表的字典
    """
    if not isinstance(keyword, str):
        raise BookAPIError("关键词格式不正确")
    keyword = keyword.strip()
    if not keyword:
        raise BookAPIError("关键词不能为空")

    if len(keyword) > 100:
        raise BookAPIError("关键词长度不能超过100个字符")

    if not type(page) is int:
        raise BookAPIError("page必须为整数")
    if page < 1:
        raise BookAPIError("page必须大于等于1")

    if not type(page_size) is int:
        raise BookAPIError("page_size必须为整数")
    if page_size < 1:
        raise BookAPIError("page_size必须大于等于1")
    if page_size > 10:
        raise BookAPIError("page_size不能超过10")

    # 规范化 bookstore_api_base_url，避免末尾斜杠导致 //book/search
    base_url = settings.bookstore_api_base_url.rstrip("/")

    # 捕获并转换 HTTPX 异常为 BookAPIError
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.get(
                base_url+"/book/search",
                params={
                    "q": keyword,
                    "page": page,
                    "page_size": page_size,
                },
            )
        # 检查异常
        response.raise_for_status()
        # 解析 JSON
        payload = response.json()
        if not isinstance(payload,dict):
            raise BookAPIError("payload格式非法")
        if payload.get("code") != 0:
            raise BookAPIError(payload.get("message", "图书搜索失败"))
    except httpx.TimeoutException:
        raise BookAPIError("书城接口请求超时")
    except httpx.RequestError:
        raise BookAPIError("无法连接 Go 书城服务")
    except httpx.HTTPStatusError as e:
        raise BookAPIError(f"Go 接口返回错误状态: {e.response.status_code}")
    except ValueError:
        raise BookAPIError("Go 接口返回的数据不是合法 JSON")

    # 验证 payload、data、books 的实际类型，避免响应异常时抛出 KeyError
    data = payload.get("data")
    if not isinstance(data, dict):
        raise BookAPIError("接口返回数据格式异常")

    books = data.get("books")
    if books is None:
        books = []
    elif not isinstance(books, list):
        raise BookAPIError("图书接口的 books 不是列表")

    # 裁剪每本书的字段，仅保留推荐所需数据
    trimmed_books = []

    for index, book in enumerate(books):
        if not isinstance(book, dict):
            raise BookAPIError(f"第{index+1}本书的格式不正确")
        book_id = book.get("id")
        title = book.get("title")
        price = book.get("price")

        if not isinstance(book_id, int):
            raise BookAPIError("图书 id 格式不正确")

        if not isinstance(title, str) or not title.strip():
            raise BookAPIError("图书标题格式不正确")

        if not type(price) is int:
            raise BookAPIError("图书价格格式不正确")

        author = book.get("author")
        if not isinstance(author, str):
            author = "作者未知"

        description = book.get("description")
        if not isinstance(description, str):
            description = ""
        description = description[:200]

        stock = book.get("stock")
        if not isinstance(stock, int):
            stock = 0

        discount = book.get("discount")
        if not isinstance(discount, int):
            discount = 0
        discount = max(0, min(discount, 100))

        book_type = book.get("type")
        if not isinstance(book_type, str):
            book_type = ""

        cover_url = book.get("cover_url")
        if not isinstance(cover_url, str):
            cover_url = ""

        current_price = price * (100 - discount) // 100

        trimmed_books.append({
            "id": book_id,
            "title": title,
            "author": author,
            "price": price,
            "discount": discount,
            "current_price": current_price,
            "type": book_type,
            "stock": stock,
            "description": description,
            "cover_url": cover_url,
        })

    total = data.get("total")
    result_page = data.get("page")
    result_page_size = data.get("page_size")
    total_pages = data.get("total_pages")

    if type(total) is not int or total < 0:
        raise BookAPIError("图书接口的 total 格式不正确")
    if type(result_page) is not int or result_page < 1:
        raise BookAPIError("图书接口的 page 格式不正确")
    if type(result_page_size) is not int or not 1 <= result_page_size <= 10:
        raise BookAPIError("图书接口的 page_size 格式不正确")
    if type(total_pages) is not int or total_pages < 0:
        raise BookAPIError("图书接口的 total_pages 格式不正确")

    return {
        "ok": True,
        "books": trimmed_books,
        "total": total,
        "page": result_page,
        "page_size": result_page_size,
        "total_pages": total_pages,
    }
