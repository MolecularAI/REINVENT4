from reinvent.runmodes.scoring.run_scoring import uniquify_header


def test_uniquify_header():
    header = ["A.1", "B", "C", "A.2", "B.2", "B"]
    result = ["A", "B", "C", "A.2", "B.2", "B.3"]

    new_header = uniquify_header(header)

    assert new_header == result
