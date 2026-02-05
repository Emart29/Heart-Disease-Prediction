# run_ci.ps1
# Local "CI test" script for Heart-Disease-Prediction
# Runs Black, Flake8, and Pytest with coverage, exits on first failure

# Exit on first error
$ErrorActionPreference = "Stop"

Write-Host "`nRunning Black (code formatting check)..."
python -m black --check .
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Black formatting check failed!"
    exit $LASTEXITCODE
}

Write-Host "`nRunning Flake8 (linting check)..."
python -m flake8 .
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Flake8 linting failed!"
    exit $LASTEXITCODE
}

Write-Host "`nRunning Pytest (unit tests + coverage)..."
python -m pytest --cov=src --cov=api
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Pytest failed!"
    exit $LASTEXITCODE
}

Write-Host "`nAll checks passed! Coverage report generated."
exit 0
