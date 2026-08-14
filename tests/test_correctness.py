from src.evaluation.correctness import check_correct

def test_math_correct():
    assert check_correct("72", "72", "math") is True

def test_math_incorrect():
    assert check_correct("70", "72", "math") is False

def test_mcq_correct():
    assert check_correct("A", "A", "commonsense") is True

def test_mcq_case_insensitive():
    assert check_correct("a", "A", "logic") is True
