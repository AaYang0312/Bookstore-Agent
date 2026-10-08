# 写操作确认门单测：待确认操作存储（内存兜底）/ 消费匹配 / 过期 / 确认消息解析
import pytest

from app.tools import write_gate


@pytest.fixture(autouse=True)
def isolated_gate(monkeypatch):
    """每个用例独立：禁用 Redis（走内存兜底）、清空存储、固定会话 ID。"""
    monkeypatch.setattr(write_gate, "get_redis", lambda: None)
    write_gate._memory_store.clear()
    write_gate.set_conversation_id("conv-test")


def _make_op(**overrides):
    operation = {"tool": "create_order", "arguments": {"items": []}, "summary": {"type": "create_order"}}
    operation.update(overrides)
    return operation


def test_create_and_consume_roundtrip():
    operation = write_gate.create_pending("create_order", {"items": [{"book_id": 1}]},
                                          {"type": "create_order"})

    assert operation["op_id"] and len(operation["op_id"]) == 16
    stored = write_gate.get_pending("conv-test")
    assert stored["tool"] == "create_order"
    assert stored["arguments"] == {"items": [{"book_id": 1}]}

    consumed = write_gate.consume_pending("conv-test", operation["op_id"])
    assert consumed["op_id"] == operation["op_id"]
    assert write_gate.get_pending("conv-test") is None  # 消费后移除


def test_new_proposal_overwrites_previous():
    write_gate.create_pending("create_order", {"items": [{"book_id": 1}]}, {})
    second = write_gate.create_pending("cancel_order", {"order_id": 5}, {})

    assert write_gate.consume_pending("conv-test")["op_id"] == second["op_id"]


def test_consume_with_wrong_op_id_keeps_operation():
    operation = write_gate.create_pending("create_order", {"items": []}, {})

    assert write_gate.consume_pending("conv-test", "0" * 16) is None
    # 不匹配不消费：原操作仍在
    assert write_gate.get_pending("conv-test")["op_id"] == operation["op_id"]


def test_consume_latest_without_op_id():
    write_gate.create_pending("create_order", {"items": []}, {})
    assert write_gate.consume_pending("conv-test") is not None
    assert write_gate.get_pending("conv-test") is None


def test_pending_expires(monkeypatch):
    monkeypatch.setattr(write_gate, "PENDING_TTL_SECONDS", -1)
    write_gate.create_pending("create_order", {"items": []}, {})

    assert write_gate.get_pending("conv-test") is None
    assert write_gate.consume_pending("conv-test") is None


def test_clear_pending():
    write_gate.create_pending("create_order", {"items": []}, {})
    assert write_gate.clear_pending("conv-test") is True
    assert write_gate.clear_pending("conv-test") is False  # 无操作可清


def test_create_requires_conversation():
    write_gate.set_conversation_id(None)
    with pytest.raises(ValueError):
        write_gate.create_pending("create_order", {}, {})


# ---- 确认消息解析 ----

@pytest.mark.parametrize("text,kind", [
    ("[CONFIRM:0123456789abcdef]", "confirm"),
    ("[CONFIRM:ABCDEF0123456789]", "confirm"),   # 大写 op_id
    ("[REJECT:0123456789abcdef]", "reject"),
    ("[CONFIRM:0123456789abcdef] ", "confirm"),  # 容忍首尾空白
    ("确认", "confirm"),
    (" 确定 ", "confirm"),
    ("确认下单", "confirm"),
    ("确认取消", "confirm"),
    ("取消操作", "reject"),
    ("暂不下单", "reject"),
])
def test_confirmation_messages(text, kind):
    parsed = write_gate.parse_confirmation(text)
    assert parsed is not None and parsed[0] == kind


def test_marker_carries_op_id():
    parsed = write_gate.parse_confirmation("[CONFIRM:0123456789abcdef]")
    assert parsed == ("confirm", "0123456789abcdef")
    assert write_gate.parse_confirmation("确认") == ("confirm", None)


@pytest.mark.parametrize("text", [
    "帮我下单一本三体",
    "确认一下有没有货",          # 确认必须独立成句
    "[CONFIRM:short]",          # op_id 长度不符
    "[CONFIRM:0123456789abcdef extra]",
    "取消订单",
    "",
    None,
])
def test_non_confirmation_messages(text):
    assert write_gate.parse_confirmation(text) is None
