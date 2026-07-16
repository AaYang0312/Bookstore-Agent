# 注册工具
import json
import logging
from app.tools.book import search_books, get_book_detail, BookAPIError
from app.tools.web import search_web, WebSearchError

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "根据书名、作者或描述关键词搜索书城中的真实图书",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词",
                        "minLength": 1,
                        "maxLength": 100,
                    },
                    "page": {
                        "type": "integer",
                        "description": "页码",
                        "default": 1,
                        "minimum": 1,
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页数量",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_book_detail",
            "description": "根据书籍 id 获取对应书籍详细信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "integer",
                        "description": "书籍 id",
                        "minimum": 1,
                    }
                },
                "required": ["book_id"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网内容，帮助用户发现最近值得阅读的书籍、书评或推荐信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        }
    }
]

TOOL_MAP = {
    "search_books": search_books,
    "get_book_detail": get_book_detail,
    "search_web": search_web,
}

def execute_tool(tool_call) -> str:
    """
    执行 LLM 返回的 tool_call，返回 JSON 字符串结果

    Args:
        tool_call: OpenAI 格式的 tool_call 对象
    """
    # 1. 读取工具名称
    name = tool_call.function.name

    # 2. 检查工具是否存在
    func = TOOL_MAP.get(name)
    if not func:
        return json.dumps({"ok": False, "error": f"未知工具: {name}"}, ensure_ascii=False)

    # 3. 解析 arguments JSON
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"ok": False, "error": "工具参数不是合法 JSON"}, ensure_ascii=False)

    # 4. 确认结果是字典
    if not isinstance(args, dict):
        return json.dumps({"ok": False, "error": "工具参数必须是对象"}, ensure_ascii=False)

    # 5. 调用 handler(**args)
    try:
        result = func(**args)
    except BookAPIError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except WebSearchError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"ok": False, "error": f"参数错误: {e}"}, ensure_ascii=False)
    except Exception as e:
        logger.exception("工具 %s 执行失败", name)
        return json.dumps({"ok": False, "error": "工具执行失败，请稍后重试"}, ensure_ascii=False)

    # 6. 序列化工具结果
    return json.dumps(result, ensure_ascii=False)