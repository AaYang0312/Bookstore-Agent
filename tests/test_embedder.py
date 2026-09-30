# Embedder 单测：批量分批 / 指数退避重试 / 维度自检 / L2 归一化
import pytest

from app.rag.embedder import EmbeddingError, Embedder


class FakeEmbeddings:
    """模拟 OpenAI embeddings 接口：可注入失败次数。"""

    def __init__(self, vectors, fail_first=0, exc=Exception("boom")):
        self.vectors = vectors
        self.fail_first = fail_first
        self.exc = exc
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise self.exc
        return SimpleResponse([self.vectors[i % len(self.vectors)] for i in range(len(input))])


class SimpleResponse:
    def __init__(self, vectors):
        self.data = [SimpleItem(vec) for vec in vectors]


class SimpleItem:
    def __init__(self, embedding):
        self.embedding = embedding


def _install(monkeypatch, embedder, fake):
    """把 _get_client 换成带 fake embeddings 的最小客户端。"""

    class _C:
        embeddings = fake

    monkeypatch.setattr(embedder, "_get_client", lambda: _C())


def _patch_settings(monkeypatch, batch_size, dim):
    from app.config import settings
    monkeypatch.setattr(settings, "embedding_batch_size", batch_size)
    monkeypatch.setattr(settings, "embedding_dim", dim)


def test_embed_texts_batches(monkeypatch):
    _patch_settings(monkeypatch, batch_size=2, dim=2)
    embedder = Embedder()
    fake = FakeEmbeddings([[0.3, 0.4], [1.0, 0.0]])
    _install(monkeypatch, embedder, fake)

    vectors = embedder.embed_texts(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert fake.calls == 3  # 2+2+1 三批
    # 归一化校验：每条向量模长为 1
    for vec in vectors:
        assert pytest.approx(sum(x * x for x in vec), abs=1e-6) == 1.0


def test_embed_texts_retry_with_backoff(monkeypatch):
    _patch_settings(monkeypatch, batch_size=4, dim=2)
    embedder = Embedder()
    fake = FakeEmbeddings([[0.6, 0.8]], fail_first=2)
    _install(monkeypatch, embedder, fake)

    sleeps = []
    monkeypatch.setattr("app.rag.embedder.time.sleep", lambda s: sleeps.append(s))

    vectors = embedder.embed_texts(["hello"])

    assert len(vectors) == 1
    assert fake.calls == 3  # 失败 2 次 + 成功 1 次
    assert sleeps == [1.0, 2.0]  # 指数退避


def test_embed_texts_retry_exhausted(monkeypatch):
    _patch_settings(monkeypatch, batch_size=4, dim=2)
    embedder = Embedder()
    fake = FakeEmbeddings([[0.6, 0.8]], fail_first=99)
    _install(monkeypatch, embedder, fake)
    monkeypatch.setattr("app.rag.embedder.time.sleep", lambda s: None)

    with pytest.raises(EmbeddingError, match="已重试"):
        embedder.embed_texts(["hello"])


def test_dimension_mismatch_detected(monkeypatch):
    _patch_settings(monkeypatch, batch_size=4, dim=4)
    embedder = Embedder()
    fake = FakeEmbeddings([[0.1, 0.2]])  # 实际维度 2 ≠ 配置 4
    _install(monkeypatch, embedder, fake)

    with pytest.raises(EmbeddingError, match="维度"):
        embedder.embed_texts(["hello"])


def test_embed_query_normalizes(monkeypatch):
    _patch_settings(monkeypatch, batch_size=4, dim=2)
    embedder = Embedder()
    fake = FakeEmbeddings([[3.0, 4.0]])
    _install(monkeypatch, embedder, fake)

    vec = embedder.embed_query("科幻小说")
    assert pytest.approx(sum(x * x for x in vec), abs=1e-6) == 1.0
