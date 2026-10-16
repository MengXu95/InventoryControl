$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $figures = @(
        'architecture',
        'negotiation',
        'gp_search',
        'gp_individual',
        'timing',
        'fixed_evaluator'
    )
    foreach ($figure in $figures) {
        $path = Join-Path $PSScriptRoot "figures\gp_agent_skills\$figure.pdf"
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing figure export: $path. Generate the editable figure deck first."
        }
    }

    & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw 'The first pdflatex pass failed.'
    }
    & bibtex main
    if ($LASTEXITCODE -ne 0) {
        throw 'BibTeX failed.'
    }
    foreach ($pass in 2..3) {
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex pass $pass failed."
        }
    }
    Write-Output "Built $(Join-Path $PSScriptRoot 'main.pdf')"
}
finally {
    Pop-Location
}
