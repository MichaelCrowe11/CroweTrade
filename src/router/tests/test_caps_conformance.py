import math
from decimal import Decimal, ROUND_HALF_UP, getcontext

caps = {
	"tick_size": Decimal("0.1"),
	"lot_size": Decimal("0.001"),
	"min_notional": Decimal("10.0"),
}


def round_tick(x: Decimal, tick: Decimal) -> Decimal:
	# Quantize to the nearest tick using banker's rounding replacement
	q = (x / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
	return q * tick


def round_lot(q: Decimal, lot: Decimal) -> Decimal:
	# Floor to the lot size
	return (q / lot).to_integral_value(rounding="ROUND_FLOOR") * lot


def test_tick_lot_min_notional():
	import random

	getcontext().prec = 28
	for _ in range(1000):
		ref = Decimal(str(random.uniform(10000, 100000)))
		jitter = Decimal(str(1 + random.uniform(-0.02, 0.02)))
		limit = round_tick(ref * jitter, caps["tick_size"])
		qty = round_lot(Decimal(str(random.uniform(0.001, 5.0))), caps["lot_size"])

		assert qty * limit >= caps["min_notional"]
		# exact divisibility checks
		assert (limit / caps["tick_size"]).quantize(Decimal("1")) == (limit / caps["tick_size"]).normalize()
		assert (qty / caps["lot_size"]).quantize(Decimal("1")) == (qty / caps["lot_size"]).normalize()

