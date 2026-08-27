from .analysis import auc, build_payload, digest


def test_auc_orientation_and_ties():
    assert auc([False, False, True, True], [0.0, 1.0, 2.0, 3.0]) == 1.0
    assert auc([False, True], [1.0, 1.0]) == 0.5


def test_payload_determinism_and_balanced_counts():
    first = build_payload()
    second = build_payload()
    assert first == second
    body = dict(first)
    claimed = body.pop("content_sha256")
    assert claimed == digest(body)
    for result in first["noise_attack"]["summary"].values():
        assert result["episodes"] == 120
        assert result["in_library_episodes"] == 100


def test_truth_does_not_enter_public_action_selection():
    payload = build_payload()
    for row in payload["noise_attack"]["rows"]:
        assert row["chosen_action"].startswith("A")
        assert "truth" not in row["chosen_action"]
