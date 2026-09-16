from swallow_yolo.swallowability import classify_swallowability, load_swallowability


def test_lookup_true_false_and_unknown(tmp_path):
    path = tmp_path / "swallowability.yaml"
    path.write_text("min_confidence: 0.7\nswallowable:\n  bead: true\n  block: false\n", encoding="utf-8")
    mapping, threshold = load_swallowability(path)
    assert threshold == 0.7
    assert classify_swallowability(mapping, "bead", 0.9, threshold).text == "能吞咽"
    assert classify_swallowability(mapping, "block", 0.9, threshold).text == "不能吞咽"
    assert classify_swallowability(mapping, "other", 0.9, threshold).can_swallow is None
    assert classify_swallowability(mapping, "bead", 0.5, threshold).text == "能吞咽"
