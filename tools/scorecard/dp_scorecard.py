import json
import sys

WEIGHTS = {
	"eos_contracts": 1.6,
	"venue_correctness": 1.2,
	"risk_governors": 1.0,
	"observability": 1.0,
	"supply_chain": 1.0,
	"audit_worm": 0.8,
	"dr_bcp": 0.8,
	"feature_store_pit": 0.8,
	"execution_tca": 0.9,
	"governance": 0.9,
}

P0 = {
	"eos_contracts",
	"venue_correctness",
	"risk_governors",
	"observability",
	"supply_chain",
	"audit_worm",
	"dr_bcp",
}


def main(path: str):
	with open(path) as f:
		m = json.load(f)
	for k in P0:
		if not m.get(f"{k}_pass", False):
			print(f"P0 failed: {k}")
			return 2
	score = sum(WEIGHTS[k] * float(m.get(k, 0.0)) for k in WEIGHTS)
	wsum = sum(WEIGHTS.values())
	final = 10.0 * score / wsum
	print(f"DP_SCORE={final:.3f}")
	if final < 9.5:
		return 3
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1]))

