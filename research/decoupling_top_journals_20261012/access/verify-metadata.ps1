$taskDois = @(
 '10.1007/s11263-026-02757-8',
 '10.1109/TPAMI.2025.3604234',
 '10.1016/j.inffus.2026.104498',
 '10.1016/j.inffus.2025.103997',
 '10.1109/TIP.2026.3689415',
 '10.1109/TIP.2026.3714832',
 '10.1016/j.isprsjprs.2026.02.024',
 '10.1109/TGRS.2026.3727427',
 '10.1109/TGRS.2026.3691782',
 '10.1109/TGRS.2025.3636047',
 '10.1007/s11263-025-02540-1',
 '10.1109/TIP.2025.3551531',
 '10.1007/s11263-025-02466-8',
 '10.1109/TIP.2025.3647207',
 '10.1109/TIP.2026.3695389',
 '10.1109/TIP.2026.3696124',
 '10.1016/j.inffus.2025.103295',
 '10.1016/j.inffus.2025.103261',
 '10.1016/j.patcog.2024.110916'
)
$taskRecords = foreach ($taskDoi in $taskDois) {
 try {
  $taskMessage = (Invoke-RestMethod -Uri ('https://api.crossref.org/works/' + [Uri]::EscapeDataString($taskDoi))).message
  [pscustomobject]@{
   doi = $taskDoi
   title = $taskMessage.title
   authors = @($taskMessage.author | ForEach-Object { ($_.given + ' ' + $_.family).Trim() })
   journal = $taskMessage.'container-title'
   published = $taskMessage.published
   online = $taskMessage.'published-online'
   print = $taskMessage.'published-print'
   volume = $taskMessage.volume
   issue = $taskMessage.issue
   page = $taskMessage.page
   article_number = $taskMessage.'article-number'
   links = $taskMessage.link
   url = $taskMessage.URL
   status = 'verified'
  }
 } catch {
  [pscustomobject]@{doi=$taskDoi;status='unavailable';error=$_.Exception.Message}
 }
}
$taskRecords | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'crossref-metadata.json') -Encoding utf8
$taskRecords | Select-Object doi,title,journal,volume,page,status | ConvertTo-Json -Depth 5
