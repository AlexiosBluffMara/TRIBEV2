# MindCat Discord bot launcher — run from PowerShell in D:\TRIBEV2
# Usage: .\start_bot.ps1

Set-Location $PSScriptRoot

# Load .env
Get-Content .env | Where-Object { $_ -match '=' -and $_ -notmatch '^\s*#' -and $_.Trim() -ne '' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
}

$env:PYTHONUNBUFFERED = "1"

$python = "$PSScriptRoot\.venv\Scripts\python.exe"

Write-Host "[start_bot] Env loaded. Starting MindCat (Jemma#1566)..." -ForegroundColor Cyan
Write-Host "[start_bot] TRIBE v2 pre-warms in ~10 s. Press Ctrl+C to stop." -ForegroundColor Yellow

& $python -m bot.bot
