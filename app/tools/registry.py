# 注册工具
import json
import logging
from app.config import settings
from app.tools.book import search_books, get_book_detail, BookAPIError
from app.tools.web import search_web, WebSearchError
from app.tools.user import (
    get_user_profile,
    get_user_orders,
    get_user_favorites,
    get_browse_history,
    UserAPIError,
)
from app.tools.rag_search import semantic_search_books, rag_available, RAGSearchError
from app.tools.bookstore_api import BookstoreAPIError
from app.tools.favorite import add_favorite, remove_favorite, check_favorite
from app.tools.order import (
    create_order, cancel_order, get_order_detail,
    propose_order, propose_cancel_order,
)

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

# 语义检索工具：仅在 Milvus 与 Embedding 均已配置时注册，未配置时完全不暴露
if rag_available():
    TOOLS.append({
        "type": "function",
        "function": {
            "name": "semantic_search_books",
            "description": (
                "按自然语言语义需求在书城中找书（如'想找讲宇宙文明的科幻小说'、"
                "'适合入门的心理学书'）。模糊、描述性找书用这个；"
                "已知精确书名/作者时请改用 search_books"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言找书需求描述",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "category": {
                        "type": "string",
                        "description": "可选，限定图书分类名称",
                        "maxLength": 50,
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "可选，价格上限（元）",
                        "minimum": 0,
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        }
    })

# 用户数据工具：随用户模块一并注册（无 JWT 时返回结构化降级提示而非报错）
TOOLS.extend([
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "获取当前登录用户的个人资料（用户名、邮箱等基本信息）",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": "获取当前登录用户最近的订单列表（含订单内图书、金额与状态）",
            "parameters": {
                "type": "object",
                "properties": {
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
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_favorites",
            "description": "获取当前登录用户收藏的图书列表",
            "parameters": {
                "type": "object",
                "properties": {
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
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_browse_history",
            "description": "获取当前登录用户最近的图书浏览记录",
            "parameters": {
                "type": "object",
                "properties": {
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
                "additionalProperties": False,
            }
        }
    },
])

TOOLS.extend([
    {
        "type": "function",
        "function": {
            "name": "add_favorite",
            "description": "把一本书加入当前用户的收藏（真实写操作）。仅在用户明确表达收藏意图时调用，book_id 必须来自检索结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "integer",
                        "description": "书籍 id（来自检索结果）",
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
            "name": "remove_favorite",
            "description": "移除当前用户对某本书的收藏（真实写操作）。仅在用户明确表达取消收藏意图时调用",
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
            "name": "check_favorite",
            "description": "查询当前用户是否已收藏某本书",
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
            "name": "propose_order",
            "description": (
                "提议为当前用户创建待支付订单（生成确认卡片，不直接下单）。"
                "用户同意下单后调用；系统会把书目/数量/预估总价卡片推给用户，"
                "用户点击确认后自动执行。book_id 必须来自检索结果"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "订单项列表",
                        "minItems": 1,
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "book_id": {
                                    "type": "integer",
                                    "description": "书籍 id（来自检索结果）",
                                    "minimum": 1,
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "购买数量",
                                    "minimum": 1,
                                    "maximum": 99,
                                }
                            },
                            "required": ["book_id", "quantity"],
                            "additionalProperties": False,
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_cancel_order",
            "description": (
                "提议取消当前用户的待支付订单（生成确认卡片，不直接取消）。"
                "用户要求取消订单时调用；用户点击确认后系统自动执行"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单 id（来自订单列表/详情）",
                        "minimum": 1,
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_detail",
            "description": "获取当前用户单笔订单的详情（含订单内图书、金额与状态），用于回答订单内容/进度类问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单 id",
                        "minimum": 1,
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            }
        }
    },
])

TOOL_MAP = {
    "search_books": search_books,
    "get_book_detail": get_book_detail,
    "search_web": search_web,
    "semantic_search_books": semantic_search_books,
    "get_user_profile": get_user_profile,
    "get_user_orders": get_user_orders,
    "get_user_favorites": get_user_favorites,
    "get_browse_history": get_browse_history,
    "add_favorite": add_favorite,
    "remove_favorite": remove_favorite,
    "check_favorite": check_favorite,
    # 写操作：LLM 只见 propose_*（生成确认卡片）；真实执行器仅供服务端确认后调用
    "propose_order": propose_order,
    "propose_cancel_order": propose_cancel_order,
    "create_order": create_order,
    "cancel_order": cancel_order,
    "get_order_detail": get_order_detail,
}

def execute_tool(name: str, arguments: str) -> str:
    """
    执行 LLM 返回的工具调用，返回 JSON 字符串结果

    Args:
        name: 工具名称
        arguments: 工具参数（JSON 字符串）
    """
    # 1. 检查工具是否存在
    func = TOOL_MAP.get(name)
    if not func:
        return json.dumps({"ok": False, "error": f"未知工具: {name}"}, ensure_ascii=False)

    # 2. 解析 arguments JSON
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"ok": False, "error": "工具参数不是合法 JSON"}, ensure_ascii=False)

    # 3. 确认结果是字典
    if not isinstance(args, dict):
        return json.dumps({"ok": False, "error": "工具参数必须是对象"}, ensure_ascii=False)

    # 4. 调用 handler(**args)
    try:
        result = func(**args)
    except BookAPIError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except WebSearchError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except UserAPIError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except RAGSearchError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except BookstoreAPIError as e:
        # 收藏/订单等走共享层的工具：message 已面向 LLM，直接透出
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"ok": False, "error": f"参数错误: {e}"}, ensure_ascii=False)
    except Exception as e:
        logger.exception("工具 %s 执行失败", name)
        return json.dumps({"ok": False, "error": "工具执行失败，请稍后重试"}, ensure_ascii=False)

    # 5. 序列化工具结果
    return json.dumps(result, ensure_ascii=False)