import pytest

from docpipe import config
from docpipe.metadata.__main__ import doc_language, doc_type, entities_from_text, locality
from docpipe.metadata.language import label
from docpipe.normalise.clean import InvariantError, check_invariant
from docpipe.scripts import share
from docpipe.triage.__main__ import classify, stray_count


def test_config_requires_ocr_lang_and_known_scripts(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("project: {dir: w}\n")
    with pytest.raises(SystemExit, match="languages.ocr"):
        config.load(f)
    f.write_text("project: {dir: w}\nlanguages: {ocr: eng, scripts: [latin, klingon]}\n")
    with pytest.raises(SystemExit, match="klingon"):
        config.load(f)
    f.write_text("project: {dir: w}\nlanguages: {ocr: eng+tel, scripts: [latin, telugu]}\n")
    c = config.load(f)
    assert c["ocr"]["tesseract"]["lang"] == "eng+tel" and c["triage"]["stray_glyphs"] == ""


def test_script_label_generalises_to_any_script():
    cfg = dict(min_chars=5, primary_below=0.02, secondary_above=0.5, _languages=dict(scripts=["latin", "cyrillic"]))
    assert share("Привет мир", "cyrillic")[0] == 1.0
    assert label({"cyrillic": 0.0}, 50, cfg) == "latin"
    assert label({"cyrillic": 0.9}, 50, cfg) == "cyrillic"
    assert label({"cyrillic": 0.2}, 50, cfg) == "mixed"
    assert label({"cyrillic": 0.9}, 2, cfg) == "unknown"


def test_triage_stray_glyphs_are_config_not_code():
    assert stray_count("a{b}", "") == 0 and stray_count("a{b}", "{}") == 2
    cfg = dict(min_chars=20, review_chars=200, image_scanned_cov=.3, image_hybrid_cov=.8, garbage_max=.02, stray_max=1, control_max=2)
    assert classify(500, 0.0, 0.0, 0, cfg)[0] == "born_digital"
    assert classify(500, 0.0, 0.0, 3, cfg)[0] == "scanned"


def test_invariant_blocks_changed_numbers():
    keep = "%.,"
    check_invariant("a 4.5 b", [], "a 4.5 b", "x", keep)
    with pytest.raises(InvariantError):
        check_invariant("a 4.5 b", [], "a 4.6 b", "x", keep)


def test_metadata_helpers_are_config_driven():
    ents, cnt = entities_from_text("wheat wheat barley " * 10, {"wheat": ["wheat"], "barley": ["barley"]}, None, 5, 0.05)
    assert ents == ["barley", "wheat"] and cnt["wheat"] == 20
    assert entities_from_text("wheat", {}, None, 1, 0)[0] is None
    cfg = dict(locality_ignore=["Kerala"], locality_title_pattern=r"(\w+) Block")
    assert locality("Kerala", "North Block plan", cfg) == ("North", "title") and locality("Idukki", "", cfg) == ("Idukki", "manifest")
    assert doc_type("A report", "", 3, dict(doc_type_rules=[{"name": "r", "title": "report"}], short_doc=None))[0] == "r"
    assert doc_language(["en"] * 9 + ["fr"], "en", .95, .5)[0] == "mixed" and doc_language(["fr"] * 9 + ["en"], "en", .95, .5)[0] == "fr"
