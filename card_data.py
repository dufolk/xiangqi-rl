"""Standard 2014 base-game Splendor development-card data (40/30/20)."""

# Entries: tier, bonus, points, then costs in white/blue/green/red/black order.
_RAW_CARDS = """
1,black,0,1,1,1,1,0
1,black,0,1,1,1,2,0
1,black,0,2,0,1,2,0
1,black,0,0,1,0,3,1
1,black,0,0,0,2,1,0
1,black,0,2,0,2,0,0
1,black,0,0,0,3,0,0
1,black,1,0,4,0,0,0
1,blue,0,1,0,1,1,1
1,blue,0,1,0,1,2,1
1,blue,0,1,0,2,2,0
1,blue,0,0,1,3,1,0
1,blue,0,1,2,0,0,0
1,blue,0,2,0,2,0,0
1,blue,0,0,3,0,0,0
1,blue,1,0,0,0,4,0
1,white,0,0,1,1,1,1
1,white,0,0,1,2,1,1
1,white,0,0,2,2,0,1
1,white,0,0,1,1,0,3
1,white,0,0,1,0,2,0
1,white,0,2,2,0,0,0
1,white,0,0,0,3,0,0
1,white,1,0,0,4,0,0
1,green,0,1,1,0,1,1
1,green,0,1,1,0,1,2
1,green,0,2,1,0,2,0
1,green,0,1,0,3,1,0
1,green,0,0,0,1,0,2
1,green,0,0,2,0,2,0
1,green,0,0,0,0,3,0
1,green,1,0,0,0,0,4
1,red,0,1,1,1,0,1
1,red,0,2,1,1,0,1
1,red,0,2,0,1,0,2
1,red,0,1,3,0,0,1
1,red,0,0,0,2,1,0
1,red,0,0,2,0,0,2
1,red,0,0,0,0,0,3
1,red,1,0,0,0,0,4
2,black,1,3,2,2,0,0
2,black,1,3,2,0,0,2
2,black,2,0,1,4,2,0
2,black,2,0,0,5,3,0
2,black,2,5,0,0,0,0
2,black,3,0,0,0,0,6
2,blue,1,0,2,2,3,0
2,blue,1,0,3,2,0,3
2,blue,2,5,0,3,0,0
2,blue,2,2,4,0,1,0
2,blue,2,0,0,5,0,0
2,blue,3,0,0,0,6,0
2,white,1,0,0,3,2,2
2,white,1,2,0,0,3,3
2,white,2,0,0,1,4,2
2,white,2,0,3,0,5,0
2,white,2,0,0,0,0,5
2,white,3,0,0,0,0,6
2,green,1,3,0,0,3,2
2,green,1,2,3,0,0,2
2,green,2,0,2,0,0,4
2,green,2,0,0,5,3,0
2,green,2,0,5,0,0,0
2,green,3,0,0,6,0,0
2,red,1,2,0,0,2,3
2,red,1,2,0,3,0,3
2,red,2,1,0,0,4,2
2,red,2,0,0,0,3,5
2,red,2,0,0,0,0,5
2,red,3,0,0,0,6,0
3,black,3,3,3,5,3,0
3,black,4,0,0,0,7,0
3,black,4,3,0,0,6,3
3,black,5,3,0,0,7,0
3,blue,3,3,0,3,3,5
3,blue,4,7,0,0,0,0
3,blue,4,6,3,0,0,3
3,blue,5,7,0,0,0,3
3,white,3,0,3,3,3,5
3,white,4,0,0,0,0,7
3,white,4,3,6,0,0,3
3,white,5,0,7,0,0,3
3,green,3,5,3,0,3,3
3,green,4,0,7,0,0,0
3,green,4,3,3,6,0,0
3,green,5,0,0,7,3,0
3,red,3,3,3,3,0,5
3,red,4,0,0,7,0,0
3,red,4,3,0,3,6,0
3,red,5,0,3,7,0,0
""".strip()

_COST_ORDER = ("white", "blue", "green", "red", "black")


def standard_card_rows():
    rows = []
    for line in _RAW_CARDS.splitlines():
        tier, bonus, points, *costs = line.split(",")
        rows.append({
            "tier": int(tier),
            "bonus": bonus,
            "points": int(points),
            "cost": dict(zip(_COST_ORDER, map(int, costs))),
        })
    return rows


STANDARD_CARD_ROWS = standard_card_rows()
assert len(STANDARD_CARD_ROWS) == 90
assert {tier: sum(row["tier"] == tier for row in STANDARD_CARD_ROWS) for tier in (1, 2, 3)} == {1: 40, 2: 30, 3: 20}
assert {(tier, bonus): sum(row["tier"] == tier and row["bonus"] == bonus for row in STANDARD_CARD_ROWS) for tier in (1, 2, 3) for bonus in _COST_ORDER} == {(tier, bonus): {1: 8, 2: 6, 3: 4}[tier] for tier in (1, 2, 3) for bonus in _COST_ORDER}
