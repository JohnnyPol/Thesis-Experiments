$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $RepoRoot

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot

$ResultsDir = if ($args.Count -ge 1) { $args[0] } else { "results/exp2_multi_model" }
$OutputDir = if ($args.Count -ge 2) { $args[1] } else { "results/thesis_visualizations/exp2_multi_model" }

Write-Host "[generate_exp2_thesis_artifacts] repo_root=$RepoRoot"
Write-Host "[generate_exp2_thesis_artifacts] results_dir=$ResultsDir"
Write-Host "[generate_exp2_thesis_artifacts] output_dir=$OutputDir"

python -m src.visualization.summary `
  --results-dir $ResultsDir `
  --output-dir $OutputDir

python -m src.visualization.tables `
  --results-dir $ResultsDir `
  --output-dir $OutputDir

python -m src.visualization.plots `
  --results-dir $ResultsDir `
  --output-dir $OutputDir

Write-Host "[generate_exp2_thesis_artifacts] done"
