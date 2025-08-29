"""
Register a dataset to OpenML with sensible local/dev defaults.

Features
- Reads a dataframe from artifacts/pit_sample.(parquet|csv) by default
- Can generate a synthetic PIT-like sample via --create-sample
- Dry-run mode to validate without publishing or requiring API keys
- Helpful errors and resilient Parquet/CSV handling

Usage examples
  python tools/openml/register_dataset.py --dry-run --create-sample
  OPENML_API_KEY=... python tools/openml/register_dataset.py \
	--df artifacts/pit_sample.parquet --name dp_crypto_h1_direction_v1
"""

from __future__ import annotations

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import click


ARTIFACTS_DIR = Path("artifacts")
DEFAULT_DF = ARTIFACTS_DIR / "pit_sample.parquet"
DEFAULT_NAME = "dp_crypto_h1_direction_v1"
DEFAULT_DESC = "PIT features for 1h direction; crypto, de-identified."


def _try_imports():
	try:
		import pandas as pd  # noqa: F401
	except Exception as e:
		raise SystemExit(
			"pandas not available. Install with: pip install pandas pyarrow"
		) from e

	try:
		import openml  # noqa: F401
	except Exception as e:
		raise SystemExit(
			"openml not available. Install with: pip install openml"
		) from e


def _read_df(df_path: Path):
	import pandas as pd
	if not df_path.exists():
		raise FileNotFoundError(f"Data file not found: {df_path}")

	try:
		if df_path.suffix.lower() in {".parquet", ".pq"}:
			return pd.read_parquet(df_path)
	except Exception as e:
		click.echo(f"Parquet read failed ({e}); attempting CSV fallback...", err=True)

	if df_path.suffix.lower() == ".csv":
		return pd.read_csv(df_path)

	# Fallback: try same stem as CSV
	csv_alt = df_path.with_suffix(".csv")
	if csv_alt.exists():
		return pd.read_csv(csv_alt)

	# Last resort: try pandas read (may guess engine)
	return pd.read_parquet(df_path)


def _write_df(df, df_path: Path):
	df_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		if df_path.suffix.lower() in {".parquet", ".pq"}:
			df.to_parquet(df_path, index=False)
			return df_path
	except Exception as e:
		click.echo(f"Parquet write failed ({e}); writing CSV fallback...", err=True)

	csv_path = df_path.with_suffix(".csv")
	df.to_csv(csv_path, index=False)
	return csv_path


def _make_sample_df(n: int = 5000):
	import pandas as pd
	import numpy as np
	rng = np.random.default_rng(42)
	# Timestamps hourly
	ts = pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC").tz_convert(None)
	# Simple synthetic features
	f1 = rng.normal(0, 1, n)
	f2 = rng.normal(0, 1, n)
	f3 = rng.lognormal(mean=0.0, sigma=0.5, size=n)
	f4 = np.tanh(f1) + 0.1 * f2 + rng.normal(0, 0.05, n)
	# Directional label derived from noisy linear combo
	raw = 0.6 * f1 - 0.4 * f2 + 0.2 * f3 + 0.05 * rng.normal(0, 1, n)
	y_dir = np.where(raw > 0.15, 1, np.where(raw < -0.15, -1, 0)).astype(int)
	df = pd.DataFrame(
		{
			"event_ts": ts,
			"entity_id": ["BTCUSDT"] * n,
			"f1": f1,
			"f2": f2,
			"f3": f3,
			"f4": f4,
			"y_dir": y_dir,
		}
	)
	return df


@click.command()
@click.option("--df", "df_path", type=click.Path(path_type=Path), default=DEFAULT_DF, help="Input DataFrame path (.parquet or .csv)")
@click.option("--name", "name", default=DEFAULT_NAME, help="OpenML dataset name")
@click.option("--desc", "desc", default=DEFAULT_DESC, help="OpenML dataset description")
@click.option("--create-sample/--no-create-sample", default=False, help="Generate a synthetic sample if df is missing")
@click.option("--dry-run/--no-dry-run", default=False, help="Validate only; do not publish to OpenML")
def main(df_path: Path, name: str, desc: str, create_sample: bool, dry_run: bool):
	"""Register dataset to OpenML or perform a local validation/dry-run."""
	_try_imports()
	import openml

	api_key = os.environ.get("OPENML_API_KEY")
	if api_key:
		openml.config.apikey = api_key

	# Prepare data
	if not Path(df_path).exists():
		if create_sample:
			click.echo(f"Input not found; creating sample at {df_path}...")
			df = _make_sample_df()
			written = _write_df(df, df_path)
			click.echo(f"Sample written: {written}")
		else:
			raise SystemExit(
				f"Data file not found: {df_path}. Use --create-sample or provide --df."
			)

	df = _read_df(df_path)
	if "y_dir" not in df.columns:
		raise SystemExit("Column 'y_dir' not found in dataframe. Add it or map your target via preprocessing.")

	ignore_cols = [c for c in ["entity_id", "event_ts"] if c in df.columns]
	click.echo(json.dumps({
		"rows": len(df),
		"cols": len(df.columns),
		"target": "y_dir",
		"ignore": ignore_cols,
		"name": name,
		"desc": desc[:80] + ("..." if len(desc) > 80 else "")
	}, indent=2))

	if dry_run:
		click.echo("Dry-run: skipping OpenML publish.")
		return 0

	if not api_key:
		raise SystemExit("OPENML_API_KEY not set and not in --dry-run. Set it to publish.")

	# Publish
	dataset = openml.datasets.create_dataset(
		name=name,
		description=desc,
		creator="Deep Parallel Genesis",
		default_target_attribute="y_dir",
		ignore_attribute=ignore_cols,
		data=df,
	)
	click.echo("Publishing dataset to OpenML...")
	dataset.publish()
	click.echo(f"OPENML_DATASET_ID={dataset.dataset_id}")


if __name__ == "__main__":
	sys.exit(main())
