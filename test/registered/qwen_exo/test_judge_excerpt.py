from types import SimpleNamespace

from qwen_exo_booster.tensor_bank import TensorBank


class _WordTokenizer:
    def __init__(self):
        self._vocab: dict[int, str] = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in str(text).split():
            token = abs(hash(word)) % 100000
            self._vocab[token] = word
            ids.append(token)
        return ids

    def decode(self, token_ids, **_kwargs):
        return " ".join(self._vocab[int(token)] for token in token_ids)


def _bank(tmp_path):
    return TensorBank(
        tmp_path / "tensor-bank.pt",
        runner=SimpleNamespace(),
        tokenizer=_WordTokenizer(),
        repositories={},
        model_fingerprint="model-fingerprint",
        max_document_tokens=512,
        salient_token_budget=64,
    )


def test_judge_excerpt_decodes_only_salient_spans(tmp_path):
    bank = _bank(tmp_path)
    words = [f"w{i}" for i in range(200)]
    words[40] = "flag{early}"
    words[150] = "flag{late}"
    candidate = SimpleNamespace(
        reference_digest="digest-1", reference_content=" ".join(words)
    )
    page = SimpleNamespace(
        page_id=3,
        lane="knowledge",
        cognition_token_count=0,
        relative_path="ctf/doc.md",
        salient_positions=(38, 39, 40, 41, 148, 149, 150, 151),
    )

    excerpt = bank._judge_excerpt(candidate, page)

    assert excerpt is not None
    assert "ctf/doc.md" in excerpt
    assert "flag{early}" in excerpt
    assert "flag{late}" in excerpt
    assert "w100" not in excerpt


def test_judge_excerpt_skips_cognition_prefix_and_caches(tmp_path):
    bank = _bank(tmp_path)
    words = [f"c{i}" for i in range(10)] + [f"d{i}" for i in range(50)]
    words[10 + 20] = "flag{mid}"
    candidate = SimpleNamespace(
        reference_digest="digest-2", reference_content=" ".join(words[10:])
    )
    page = SimpleNamespace(
        page_id=4,
        lane="knowledge",
        cognition_token_count=10,
        relative_path="ctf/doc2.md",
        salient_positions=(0, 5, 30),
    )

    excerpt = bank._judge_excerpt(candidate, page)

    assert excerpt is not None
    assert "flag{mid}" in excerpt
    assert "c0" not in excerpt
    assert bank._judge_excerpt(candidate, page) == excerpt


def test_judge_excerpt_ignores_non_knowledge_lanes(tmp_path):
    bank = _bank(tmp_path)
    candidate = SimpleNamespace(reference_digest="digest-3", reference_content="text")
    page = SimpleNamespace(
        page_id=5,
        lane="cognition",
        cognition_token_count=0,
        relative_path="card.md",
        salient_positions=(0,),
    )
    assert bank._judge_excerpt(candidate, page) is None
