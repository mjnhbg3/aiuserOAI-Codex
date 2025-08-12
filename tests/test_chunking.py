from gpt5assistant.utils.chunking import chunk_message


def test_chunking_closes_code_fences():
    text = """Here is code:\n```python\nprint('hi')\n"""
    parts = chunk_message(text, limit=50)
    assert len(parts) >= 1
    assert parts[0].endswith("```")


def test_chunking_respects_limit():
    text = "x" * 5000
    parts = chunk_message(text)
    assert all(len(p) <= 2000 for p in parts)
    assert sum(len(p) for p in parts) >= 5000 - len(parts)  # allow fence closures

