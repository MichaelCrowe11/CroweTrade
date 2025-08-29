from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime
from bisect import bisect_right


@given(
	st.lists(
		st.tuples(
			st.datetimes(min_value=datetime(2017, 1, 1), max_value=datetime(2025, 1, 1)),
			st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False),
		),
		min_size=50,
		max_size=200,
	)
)
@settings(max_examples=5, suppress_health_check=[HealthCheck.large_base_example])
def test_pit_no_future(samples):
	# Pseudocode: ensure PIT builder never reads labels (t+Δ)
	# Replace with your actual feature builder and labeler
	samples = sorted(samples)

	def build_features_asof(t):
		return [x for ts, x in samples if ts <= t]

	times = [ts for ts, _ in samples]
	for i, (ts, _) in enumerate(samples[:-10]):
		feats = build_features_asof(ts)
		cutoff = bisect_right(times, ts)  # include ties at ts
		# No future leakage: exactly count observations with ts <= t
		assert len(feats) == cutoff

