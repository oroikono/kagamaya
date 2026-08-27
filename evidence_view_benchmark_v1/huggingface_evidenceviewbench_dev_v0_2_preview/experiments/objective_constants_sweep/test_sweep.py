from sweep import margin, objective


def row(status: str, residual: float = 0.01, gap: float = 0.02, condition: float = 2.0):
    return {
        "terminal_status": status,
        "best_residual": residual,
        "score_gap": gap,
        "best_condition": condition,
    }


def test_frozen_identified_margin():
    assert abs(margin(row("IDENTIFIED"), 0.05, 0.20) - 0.8) < 1e-12


def test_frozen_omitted_margin():
    assert abs(margin(row("OUTSIDE_LIBRARY", residual=0.4), 0.05, 0.20) - 1.0) < 1e-12


def test_inconclusive_uses_larger_hypothesis_margin():
    item = row("INCONCLUSIVE", residual=0.4)
    assert margin(item, 0.05, 0.20) == margin(row("OUTSIDE_LIBRARY", residual=0.4), 0.05, 0.20)


def test_terminal_bonus_is_preserved():
    assert objective(row("IDENTIFIED"), 7, 0.05, 0.20) > objective(row("INCONCLUSIVE"), 7, 0.05, 0.20)
