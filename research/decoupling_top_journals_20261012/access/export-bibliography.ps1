$taskRecords = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'crossref-metadata.json') -Raw | ConvertFrom-Json
$taskKeys = @('MossFuse2026','NudPaf2025','RasdFuNet2026','CoPGNet2026','VPGNet2026','BHSRNet2026','SpecAware2026','CrossSensor2026')
$taskEntries = for ($taskIndex=0; $taskIndex -lt $taskKeys.Count; $taskIndex++) {
 $taskRecord = $taskRecords[$taskIndex]
 $taskYear = $taskRecord.published.'date-parts'[0][0]
 $taskAuthor = $taskRecord.authors -join ' and '
 $taskTitle = [System.Net.WebUtility]::HtmlDecode($taskRecord.title[0])
 $taskPage = $taskRecord.page -replace '-', '--'
 $taskFields = @(
  ('@article{' + $taskKeys[$taskIndex] + ','),
  ('  title = {{' + $taskTitle + '}},'),
  ('  author = {' + $taskAuthor + '},'),
  ('  journal = {' + $taskRecord.journal[0] + '},'),
  ('  year = {' + $taskYear + '},'),
  ('  volume = {' + $taskRecord.volume + '},')
 )
 if ($taskRecord.issue) {$taskFields += ('  number = {' + $taskRecord.issue + '},')}
 if ($taskRecord.article_number) {
  $taskFields += ('  eid = {' + $taskRecord.article_number + '},')
 } elseif ($taskRecord.page -match '^(\d+)-\1$') {
  $taskFields += ('  eid = {' + $Matches[1] + '},')
 } elseif ($taskPage) {
  $taskFields += ('  pages = {' + $taskPage + '},')
 }
 if ($taskKeys[$taskIndex] -eq 'RasdFuNet2026') {
  $taskFields += '  note = {Online material accessed 2026-10-12; assigned to the December 2026 issue; exact first-online date not verified},'
 }
 $taskFields += ('  doi = {' + $taskRecord.doi + '},')
 $taskFields += ('  url = {https://doi.org/' + $taskRecord.doi + '}')
 $taskFields += '}'
 $taskFields -join "`n"
}
($taskEntries -join "`n`n") | Set-Content -LiteralPath (Join-Path $PSScriptRoot '..\references.bib') -Encoding utf8
'Exported 8 verified DOI records.'
