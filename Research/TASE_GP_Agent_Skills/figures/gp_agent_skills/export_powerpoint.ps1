param(
    [Parameter(Mandatory = $true)][string]$DeckPath,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$names = @('architecture', 'negotiation', 'gp_search', 'gp_individual', 'timing', 'fixed_evaluator')
$app = $null
$presentation = $null
try {
    $app = New-Object -ComObject PowerPoint.Application
    for ($index = 1; $index -le $names.Count; $index++) {
        $presentation = $app.Presentations.Open($DeckPath, -1, -1, 0)
        if ($presentation.Slides.Count -ne $names.Count) {
            throw "Expected $($names.Count) slides."
        }
        for ($slide = $presentation.Slides.Count; $slide -ge 1; $slide--) {
            if ($slide -ne $index) { $presentation.Slides.Item($slide).Delete() }
        }
        $destination = Join-Path $OutputDirectory ($names[$index - 1] + '.pdf')
        $presentation.SaveAs($destination, 32)
        $presentation.Saved = -1
        $presentation.Close()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
        $presentation = $null
        Write-Output ("Exported " + $destination)
    }
}
finally {
    if ($presentation) {
        $presentation.Saved = -1
        $presentation.Close()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    }
    if ($app) {
        # PowerPoint may reuse a user's server; release COM without quitting it.
        [Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
    }
}
