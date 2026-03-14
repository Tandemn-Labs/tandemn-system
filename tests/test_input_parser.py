from orca_server.input_parser import estimate_tokens, extract_prompt_text


def test_estimate_tokens_basic():
    assert estimate_tokens("hello world") == max(1, len("hello world") // 4)


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 1


def test_extract_prompt_text_valid():
    entry = {
        "body": {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"},
            ]
        }
    }
    text = extract_prompt_text(entry)
    assert "Hello" in text
    assert "World" in text


def test_extract_prompt_text_missing():
    assert extract_prompt_text({}) == ""
    assert extract_prompt_text({"body": {}}) == ""
