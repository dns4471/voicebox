from backend.routes.generations import _merge_voice_attrs_into_instruct, _parse_voice_block


def test_parse_voice_block_extracts_supported_attrs_and_text():
    text, attrs = _parse_voice_block('<v emotion="angry" pace="fast" tone="sharp" volume="loud" action="shout">안녕하세요</v>')
    assert text == "안녕하세요"
    assert attrs == {
        "emotion": "angry",
        "pace": "fast",
        "tone": "sharp",
        "volume": "loud",
        "action": "shout",
    }


def test_parse_voice_block_keeps_plain_text_unchanged():
    src = "그냥 일반 텍스트"
    text, attrs = _parse_voice_block(src)
    assert text == src
    assert attrs == {}


def test_merge_voice_attrs_into_existing_instruct():
    merged = _merge_voice_attrs_into_instruct("Speak clearly", {"emotion": "angry", "pace": "fast"})
    assert merged == "Speak clearly\nVoice style: emotion: angry, pace: fast"
