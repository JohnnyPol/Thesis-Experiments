$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $RepoRoot

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot

$ResultsDir = if ($args.Count -ge 1) { $args[0] } else { "results/exp3_memory_aware_multi_model" }
$OutputDir = if ($args.Count -ge 2) { $args[1] } else { "results/thesis_visualizations/exp3_memory_aware_multi_model" }

Write-Host "[generate_exp3_thesis_artifacts] repo_root=$RepoRoot"
Write-Host "[generate_exp3_thesis_artifacts] results_dir=$ResultsDir"
Write-Host "[generate_exp3_thesis_artifacts] output_dir=$OutputDir"

function Invoke-PythonStep {
  param([string[]]$Arguments)
  python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
  }
}

Invoke-PythonStep @(
  "-m", "src.visualization.summary",
  "--results-dir", $ResultsDir,
  "--output-dir", $OutputDir
)

Invoke-PythonStep @(
  "-m", "src.visualization.tables",
  "--results-dir", $ResultsDir,
  "--output-dir", $OutputDir
)

Invoke-PythonStep @(
  "-m", "src.visualization.plots",
  "--results-dir", $ResultsDir,
  "--output-dir", $OutputDir
)

Write-Host "[generate_exp3_thesis_artifacts] done"
