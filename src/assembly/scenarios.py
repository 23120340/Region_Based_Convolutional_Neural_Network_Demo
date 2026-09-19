SCENARIOS: dict[str, tuple[str, ...]] = {
    "correct": (
        "pick_barrel",
        "insert_refill",
        "insert_spring",
        "screw_cap",
        "test_click",
    ),
    "missing_spring": (
        "pick_barrel",
        "insert_refill",
        "screw_cap",
    ),
    "missing_refill": (
        "pick_barrel",
        "screw_cap",
    ),
    "wrong_order": (
        "pick_barrel",
        "insert_spring",
    ),
    "premature_test": (
        "pick_barrel",
        "insert_refill",
        "insert_spring",
        "test_click",
    ),
}

