# 调用go图书接口
import httpx
from app.config import settings

class BookAPIError(Exception):
    """图书 API 调用异常"""
    pass

def _create_http_client(timeout: float, trust_env: bool) -> httpx.Client:
    return httpx.Client(timeout=timeout, trust_env=trust_env)

def _request_bookstore(path: str, params: dict | None = None) -> dict:
    # 规范化 bookstore_api_base_url，避免末尾斜杠导致 //book/search
    base_url = settings.bookstore_api_base_url.rstrip("/")

    # 捕获并转换 HTTPX 异常为 BookAPIError
    try:
        with _create_http_client(timeout=5.0, trust_env=False) as client:
            response = client.get(
                base_url + path,
                params=params,
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
        if e.response.status_code == 404:
            raise BookAPIError("图书不存在或已经下架")
        raise BookAPIError(f"Go 接口返回错误状态: {e.response.status_code}")
    except ValueError:
        raise BookAPIError("Go 接口返回的数据不是合法 JSON")

    return payload


def _normalize_book(book: dict, include_details: bool = False) -> dict:
    """规范化单本书数据。

    Args:
        book: 原始图书字典
        include_details: False 返回搜索摘要，True 额外返回详情字段

    Returns:
        规范化后的图书字典
    """
    if not isinstance(book, dict):
        raise BookAPIError("图书格式不正确")

    # --- 必填字段验证 ---
    book_id = book.get("id")
    if not isinstance(book_id, int):
        raise BookAPIError("图书 id 格式不正确")

    title = book.get("title")
    if not isinstance(title, str) or not title.strip():
        raise BookAPIError("图书标题格式不正确")

    price = book.get("price")
    if not type(price) is int:
        raise BookAPIError("图书价格格式不正确")

    # --- 公共可选字段 ---
    author = book.get("author")
    if not isinstance(author, str):
        author = "作者未知"

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

    description = book.get("description")
    if not isinstance(description, str):
        description = ""

    result = {
        "id": book_id,
        "title": title,
        "author": author,
        "type": book_type,
        "price": price,
        "discount": discount,
        "current_price": current_price,
        "stock": stock,
        "description": description[:200] if not include_details else description[:1000],
        "cover_url": cover_url,
    }

    # --- 详情额外字段 ---
    if include_details:
        isbn = book.get("isbn")
        result["isbn"] = isbn if isinstance(isbn, str) else ""

        publisher = book.get("publisher")
        result["publisher"] = publisher if isinstance(publisher, str) else ""

        pages = book.get("pages")
        result["pages"] = pages if isinstance(pages, int) and pages >= 0 else 0

        language = book.get("language")
        result["language"] = language if isinstance(language, str) else ""

        fmt = book.get("format")
        result["format"] = fmt if isinstance(fmt, str) else ""

    return result


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

    payload = _request_bookstore("/book/search", params={"q": keyword, "page": page, "page_size": page_size})

    # 验证 payload、data、books 的实际类型，避免响应异常时抛出 KeyError
    data = payload.get("data")
    if not isinstance(data, dict):
        raise BookAPIError("接口返回数据格式异常")
    books = data.get("books")
    if books is None:
        books = []
    elif not isinstance(books, list):
        raise BookAPIError("图书接口的 books 不是列表")

    trimmed_books = [_normalize_book(book) for book in books]

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

def get_book_detail(book_id: int) -> dict:
    """
    查看图书详情

    Args:
        book_id: 书籍 id，从 1 开始

    Returns:
        包含书籍详情的字典
    """
    if not type(book_id) is int:
        raise BookAPIError("书籍 id 必须为整数")
    if not book_id >= 1:
        raise BookAPIError("书籍 id 必须大于等于 1")

    path = "/book/detail/"+str(book_id)
    payload = _request_bookstore(path=path)

    data = payload.get("data")
    if not isinstance(data, dict):
        raise BookAPIError("接口返回数据格式异常")

    details = _normalize_book(book=data,include_details=True)
    return {"ok":True,
        "book":details,
    }