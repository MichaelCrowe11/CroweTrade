Param(
  [switch]$Force
)

Write-Host "[dev-bootstrap] Creating virtual environment (.venv)" -ForegroundColor Cyan
if (Test-Path .venv -and $Force) { Remove-Item -Recurse -Force .venv }
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
Write-Host "[dev-bootstrap] Installing project in editable mode with dev extras" -ForegroundColor Cyan
pip install -e .[dev]
Write-Host "[dev-bootstrap] Running lint + type + tests" -ForegroundColor Cyan
ruff check src || exit 1
mypy src/crowetrade || Write-Host "[dev-bootstrap] mypy completed"
pytest -q || exit 1
Write-Host "[dev-bootstrap] Complete" -ForegroundColor Green