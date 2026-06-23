from qfs.algorithms.shor import order_from_phase


def test_dyadic_phases():
    assert order_from_phase(0.75, 15) == 4    # 3/4
    assert order_from_phase(0.25, 15) == 4    # 1/4
    assert order_from_phase(0.5, 15) == 2     # 1/2
    assert order_from_phase(0.0, 15) == 1     # 0/1


def test_non_dyadic_phases():
    # the continued-fraction half is N-independent and handles approximate phases
    assert order_from_phase(0.833, 21) == 6   # approx 5/6
    assert order_from_phase(1 / 3, 21) == 3   # 1/3
