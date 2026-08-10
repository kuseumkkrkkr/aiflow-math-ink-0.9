param(
  [string]$DatasetRoot = (Join-Path $PSScriptRoot '..\datasets')
)

$ErrorActionPreference = 'Stop'
$expected = [ordered]@{
  '10_approved_external\uji_pen_characters_v2\raw\uji_pen_v2.zip' = '0881B522911B99D9922820289441B50FD3D307F71CD7F9CC70E86872424A5F90'
  '10_approved_external\uji_pen_characters_v2\derived\uji_math_curated.jsonl.gz' = 'CDA2FE17C213BC8C90ED93EC8998658DDAF9D7871C75ADF521798E6E3BA3029B'
  '20_reaudit_required\isgl_online_offline_hwr\migrated\isgl_online.jsonl.gz' = '0A4BD4FE37CD7656B2FD3586FED4D5C886FEBA1D8E5120803B60A0F04044988D'
  '20_reaudit_required\uci_character_trajectories\raw\character_trajectories.zip' = '5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52'
}

foreach ($entry in $expected.GetEnumerator()) {
  $path = Join-Path $DatasetRoot $entry.Key
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing dataset artifact: $path" }
  if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.Value) { throw "Hash mismatch: $path" }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($relative in @(
  '10_approved_external\uji_pen_characters_v2\raw\uji_pen_v2.zip',
  '20_reaudit_required\uci_character_trajectories\raw\character_trajectories.zip'
)) {
  $archive = [IO.Compression.ZipFile]::OpenRead((Join-Path $DatasetRoot $relative))
  try {
    if ($archive.Entries.Count -lt 1) { throw "Empty ZIP: $relative" }
  } finally { $archive.Dispose() }
}

foreach ($relative in @(
  '10_approved_external\uji_pen_characters_v2\derived\uji_math_curated.jsonl.gz',
  '20_reaudit_required\isgl_online_offline_hwr\migrated\isgl_online.jsonl.gz'
)) {
  $input = [IO.File]::OpenRead((Join-Path $DatasetRoot $relative))
  $gzip = [IO.Compression.GZipStream]::new($input, [IO.Compression.CompressionMode]::Decompress)
  try {
    $buffer = New-Object byte[] 65536
    while ($gzip.Read($buffer, 0, $buffer.Length) -gt 0) {}
  } finally {
    $gzip.Dispose()
    $input.Dispose()
  }
}

Write-Output "Verified $($expected.Count) dataset artifacts."
