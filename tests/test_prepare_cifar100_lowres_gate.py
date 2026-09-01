from scripts.prepare_cifar100_lowres_gate import select_by_hash


def test_select_by_hash_balances_all_fine_classes():
    rows = []
    for fine in range(100):
        for index in range(3):
            rows.append(
                {
                    "fine_label": fine,
                    "coarse_label": fine // 5,
                    "img": {"bytes": f"{fine}-{index}".encode()},
                }
            )
    selected = select_by_hash(rows, per_class=2)
    assert len(selected) == 200
    assert {row["fine_label"] for row in selected} == set(range(100))


def test_select_by_hash_is_order_independent():
    rows = [
        {"fine_label": fine, "coarse_label": fine // 5, "img": {"bytes": f"{fine}-{index}".encode()}}
        for fine in range(100)
        for index in range(2)
    ]
    forward = select_by_hash(rows, per_class=1)
    reverse = select_by_hash(list(reversed(rows)), per_class=1)
    assert [row["img"]["bytes"] for row in forward] == [row["img"]["bytes"] for row in reverse]


def test_select_by_hash_supports_disjoint_rank_windows():
    rows = [
        {
            "fine_label": fine,
            "coarse_label": fine // 5,
            "img": {"bytes": f"{fine}-{index}".encode()},
        }
        for fine in range(100)
        for index in range(4)
    ]
    first = select_by_hash(rows, per_class=2, rank_start=0)
    second = select_by_hash(rows, per_class=2, rank_start=2)
    first_hashes = {row["img"]["bytes"] for row in first}
    second_hashes = {row["img"]["bytes"] for row in second}
    assert len(first) == len(second) == 200
    assert first_hashes.isdisjoint(second_hashes)
