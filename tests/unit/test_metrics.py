from src.evaluation.metrics import exact_match, normalize_text, token_f1


def test_normalize_text():
    assert normalize_text("  Bonjour,   MONDE ! ") == "bonjour monde"


def test_exact_match():
    assert exact_match("Paris", "Paris") == 1.0
    assert exact_match("Paris", "Lyon") == 0.0


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match("La capitale est Yaoundé.", "la capitale est yaoundé") == 1.0


def test_token_f1_identical_text():
    assert token_f1("bonjour monde", "bonjour monde") == 1.0


def test_token_f1_no_overlap():
    assert token_f1("bonjour", "monde") == 0.0


def test_token_f1_partial_overlap():
    score = token_f1("bonjour monde", "bonjour Afrique")

    assert 0.0 < score < 1.0
