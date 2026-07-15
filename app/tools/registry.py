# 注册工具
import json
import logging
from app.tools.book import search_books, BookAPIError

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
    }
]

TOOL_MAP = {
    "search_books": search_books,
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
    except TypeError as e:
        return json.dumps({"ok": False, "error": f"参数错误: {e}"}, ensure_ascii=False)
    except Exception as e:
        logger.exception("工具 %s 执行失败", name)
        return json.dumps({"ok": False, "error": "工具执行失败，请稍后重试"}, ensure_ascii=False)

    # 6. 序列化工具结果
    return json.dumps(result, ensure_ascii=False)