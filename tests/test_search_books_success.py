import httpx

from app.tools import book as book_module


def test_book_search(monkeypatch):
    # 1. 模拟 Go 后端返回的数据
    payload = {
        "code": 0,
        "message": "搜索书籍成功",
        "data": {
            "books": [
                {
                    "id": 1,
                    "title": "三体",
                    "author": "刘慈欣",
                    "price": 59,
                    "discount": 20,
                    "type": "科幻",
                    "stock": 100,
                    "description": "地球文明与三体文明的故事。",
                    "cover_url": "https://example.com/santi.jpg",
                }
            ],
            "total": 1,
            "page": 1,
            "page_size": 5,
            "total_pages": 1,
        },
    }

    # 2. 收到请求返回模拟数据
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/book/search"
        assert request.url.params["q"] == "三体"
        assert request.url.params["page"] == "1"
        assert request.url.params["page_size"] == "5"

        return httpx.Response(
            status_code=200,
            json=payload
        )

    transport = httpx.MockTransport(mock_handler)

    # 3. 保存真正的 Client，避免 monkeypatch 后递归调用自己
    real_client_class = httpx.Client

    def mock_client_factory(timeout, trust_env):
        return real_client_class(
            timeout=timeout,
            trust_env=trust_env,
            transport=transport,
        )

    monkeypatch.setattr(
        book_module,
        "_create_http_client",
        mock_client_factory,
    )

    # 4. 调用真实 search_books
    result = book_module.search_books(
        keyword="三体",
        page=1,
        page_size=5,
    )

    # 5. 检查 search_books 处理后的结果
    assert result["ok"] is True
    assert result["total"] == 1
    assert result["page"] == 1
    assert result["page_size"] == 5
    assert result["total_pages"] == 1

    assert len(result["books"]) == 1

    book = result["books"][0]

    assert book["id"] == 1
    assert book["title"] == "三体"
    assert book["author"] == "刘慈欣"
    assert book["price"] == 59
    assert book["discount"] == 20
    assert book["current_price"] == 47
    assert book["stock"] == 100
