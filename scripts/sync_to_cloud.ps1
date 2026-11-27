param(
    [int]$Limit = 10,
    [switch]$SkipIngest,
    [switch]$SkipAsr,
    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# 锁定所有缓存与临时目录在 E 盘项目内，绝不占用 C 盘
$env:HF_HOME = Join-Path $Root "data\cache\huggingface"
$env:TMP = Join-Path $Root "data\tmp_asr"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TMP | Out-Null

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "未找到虚拟环境 Python: $Py"
}

# 读取或首次引导配置 .cloud_sync.env（已在 .gitignore 中排除，不上传 GitHub）
$ConfFile = Join-Path $Root ".cloud_sync.env"
$Conf = @{
    CLOUD_HOST    = ""
    CLOUD_APP_DIR = "/opt/zhijiang"
    PUBLIC_URL    = "https://zjdb.mprlzy.top"
    INGEST_LIMIT  = "$Limit"
}
if (Test-Path $ConfFile) {
    Get-Content $ConfFile -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)\s*$') {
            $Conf[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
}

if (-not $Conf["CLOUD_HOST"]) {
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host " 首次运行增量同步配置（配置将保存在本地 .cloud_sync.env）" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
    $ip = (Read-Host "请输入你的云服务器公网 IP 或 root@IP（例如 47.xx.xx.xx）").Trim()
    if (-not $ip) { throw "未输入服务器地址，已取消。" }
    if ($ip -notmatch '@') { $ip = "root@$ip" }
    $Conf["CLOUD_HOST"] = $ip
    $lines = @(
        "CLOUD_HOST=$($Conf['CLOUD_HOST'])"
        "CLOUD_APP_DIR=$($Conf['CLOUD_APP_DIR'])"
        "PUBLIC_URL=$($Conf['PUBLIC_URL'])"
        "INGEST_LIMIT=$($Conf['INGEST_LIMIT'])"
    )
    [System.IO.File]::WriteAllLines($ConfFile, $lines, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "[OK] 已保存服务器配置到 $ConfFile`n" -ForegroundColor Green
}

$CloudHost   = $Conf["CLOUD_HOST"]
$CloudAppDir = $Conf["CLOUD_APP_DIR"]
$PublicUrl   = $Conf["PUBLIC_URL"].TrimEnd("/")
if ($PSBoundParameters.ContainsKey("Limit")) {
    $EffLimit = $Limit
} elseif ($Conf["INGEST_LIMIT"]) {
    $EffLimit = [int]$Conf["INGEST_LIMIT"]
} else {
    $EffLimit = 10
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " 枝江数据库 · 本地增量更新 & 极速热同步上云" -ForegroundColor Cyan
Write-Host " 目标服务器: $CloudHost ($CloudAppDir)" -ForegroundColor Gray
Write-Host " 公开站点:   $PublicUrl" -ForegroundColor Gray
Write-Host "========================================================`n" -ForegroundColor Cyan

# 同步前先抓一眼公网首页当前数据，方便结束后对比
$BeforeStats = $null
try {
    $BeforeStats = Invoke-RestMethod -Uri "$PublicUrl/api/v1/stats" -TimeoutSec 10
    Write-Host ("[同步前公网首页] 字幕: {0:N0} 条 | 有效分P: {1:N0} | 时间跨度: {2} ~ {3}" -f `
        $BeforeStats.subtitles_total, $BeforeStats.video_parts_ok, $BeforeStats.date_min, $BeforeStats.date_max) -ForegroundColor DarkGray
} catch {
    Write-Host "[提示] 暂未读取到同步前公网统计，继续执行同步..." -ForegroundColor Yellow
}

# ── 第 1 步：本地增量追新 + 语音补录 + 词表纠错 + 场景打标 ──
if (-not $SyncOnly -and -not $SkipIngest) {
    Write-Host "`n==> [1/4] 增量抓取 6 位成员最新直播回放 AI 字幕（每人检查最近 $EffLimit 场）..." -ForegroundColor Cyan
    & $Py run.py ingest-up all --limit $EffLimit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] 抓取阶段返回非零状态（可能是 B 站 Cookie 过期，可运行 python run.py login 续期）" -ForegroundColor Yellow
    }

    if (-not $SkipAsr) {
        Write-Host "`n==> [2/4] 检查无自带字幕的新分 P，调用本地 GPU faster-whisper 自动补录..." -ForegroundColor Cyan
        & $Py run.py backfill-asr --limit 0 --batch-size 12 --workers 1 --prefetch 1
    } else {
        Write-Host "`n==> [2/4] 已按参数跳过本地 GPU 语音补录 (-SkipAsr)" -ForegroundColor DarkGray
    }

    Write-Host "`n==> [3/4] 执行枝江专有名词纠错与新视频场景初筛..." -ForegroundColor Cyan
    & $Py run.py normalize-corpus
    & $Py run.py tag-scenes --only-unknown
} else {
    Write-Host "`n==> [1-3/4] 已跳过本地抓取，直接比对并同步本地与云端的差异数据..." -ForegroundColor DarkGray
}

# ── 第 2 步：远程拉取最新代码 + 清单比对 + 生成增量补丁包（字幕 + 声纹） ──
Write-Host "`n==> [4/4] 拉取云端最新代码与数据清单，计算极速增量补丁..." -ForegroundColor Cyan
$ManifestFile = Join-Path $Root "data\tmp_asr\remote_manifest.json"
$DeltaFile    = Join-Path $Root "data\tmp_asr\zhijiang_delta.json.gz"
$SpeakersFile = Join-Path $Root "data\tmp_asr\zhijiang_speakers.json.gz"
Remove-Item -Force -ErrorAction SilentlyContinue $ManifestFile, $DeltaFile, $SpeakersFile

# 先拉取云端最新代码（保护公告文件），并在一键更新时平滑重启云端服务（-SyncOnly 纯同步模式不拉代码也不重启）
if (-not $SyncOnly) {
    $PullAndRestartCmd = @"
set -e
cd $CloudAppDir
git checkout -- config/announcements.json 2>/dev/null || true
git pull --ff-only -q origin main
SVC=`$(systemctl list-unit-files --type=service --no-legend | awk '{print `$1}' | grep -iE 'zhijiang|zjdb' | head -n 1 || true)
if [ -n "`$SVC" ]; then
  echo "[remote] 正在重启 `$SVC ..." >&2
  systemctl restart "`$SVC"
  sleep 1
  echo "[remote] 服务状态: `$(systemctl is-active "`$SVC")" >&2
fi
.venv/bin/python run.py sync-manifest
"@
    $PullAndRestartCmd = $PullAndRestartCmd -replace "`r`n", "`n"
    $RemoteManifestJson = & ssh $CloudHost $PullAndRestartCmd
} else {
    $RemoteManifestJson = & ssh $CloudHost "cd $CloudAppDir && .venv/bin/python run.py sync-manifest"
}
if ($LASTEXITCODE -ne 0) {
    throw "SSH 拉取云端清单失败，请检查网络连接或服务器密码/密钥。"
}
[System.IO.File]::WriteAllText($ManifestFile, ($RemoteManifestJson -join "`n"), (New-Object System.Text.UTF8Encoding($false)))

# 同步本地站点公告（若存在）
$AnnFile = Join-Path $Root "config\announcements.json"
if (Test-Path $AnnFile) {
    $null = Get-Content $AnnFile -Raw -Encoding UTF8 | ConvertFrom-Json
    & scp -q $AnnFile "${CloudHost}:${CloudAppDir}/config/announcements.json"
}

$ExportOut = & $Py run.py export-delta --remote-manifest $ManifestFile --out $DeltaFile
if ($LASTEXITCODE -ne 0) { throw "本地生成增量补丁失败。" }
$ExportInfo = ($ExportOut -join "`n") | ConvertFrom-Json

Write-Host ("    本地分P: {0} | 云端分P: {1} | 需同步分P: {2}（新增 {3}，更新 {4}）| 字幕: {5:N0} 条 | 补丁体积: {6} KB" -f `
    $ExportInfo.local_parts, $ExportInfo.remote_parts, $ExportInfo.parts_changed, `
    $ExportInfo.parts_new, $ExportInfo.parts_updated, $ExportInfo.subtitles_exported, $ExportInfo.delta_size_kb) -ForegroundColor White

$ApplyInfo = [pscustomobject]@{ parts_applied = 0; subtitles_inserted = 0 }
if ([int]$ExportInfo.parts_changed -gt 0) {
    # ── 第 3 步：秒传小补丁到服务器并 WAL 热写入（零停服） ──
    Write-Host "`n    正在上传增量字幕补丁 ($($ExportInfo.delta_size_kb) KB) 到 $CloudHost ..." -ForegroundColor Cyan
    & scp $DeltaFile "${CloudHost}:/tmp/zhijiang_delta.json.gz"
    if ($LASTEXITCODE -ne 0) { throw "SCP 上传补丁失败。" }

    Write-Host "    正在云端热合并入库并增量更新 FTS5 索引（不中断公网访问）..." -ForegroundColor Cyan
    $ApplyOut = & ssh $CloudHost "cd $CloudAppDir && .venv/bin/python run.py apply-delta /tmp/zhijiang_delta.json.gz && rm -f /tmp/zhijiang_delta.json.gz"
    if ($LASTEXITCODE -ne 0) { throw "云端合并补丁失败。" }
    $ApplyInfo = ($ApplyOut -join "`n") | ConvertFrom-Json
} else {
    Write-Host "    [OK] 字幕库已与云端 100% 一致，无需上传字幕补丁。" -ForegroundColor Green
}

# 检查并增量同步声纹发言人标注补丁（零 FTS5 重建）
$SpkExportOut = & $Py -c "import json; from pathlib import Path; from zhijiang_corpus.config import load_config; from zhijiang_corpus.db import CorpusDB; from zhijiang_corpus.sync_delta import export_speakers; cfg=load_config(); db=CorpusDB(cfg.db_path); rm=json.loads(Path(r'$ManifestFile').read_text(encoding='utf-8')); print(json.dumps(export_speakers(db, Path(r'$SpeakersFile'), remote_manifest=rm), ensure_ascii=False))"
if ($LASTEXITCODE -eq 0 -and $SpkExportOut) {
    $SpkInfo = ($SpkExportOut -join "`n") | ConvertFrom-Json
    if ([int]$SpkInfo.parts_exported -gt 0) {
        Write-Host ("    发现新声纹标注：{0} 个分P / {1:N0} 条发言人标签（{2} KB），正在热同步..." -f `
            $SpkInfo.parts_exported, $SpkInfo.lines_exported, $SpkInfo.patch_size_kb) -ForegroundColor Cyan
        & scp $SpeakersFile "${CloudHost}:/tmp/zhijiang_speakers.json.gz"
        if ($LASTEXITCODE -ne 0) { throw "SCP 上传声纹补丁失败。" }
        $SpkApplyOut = & ssh $CloudHost "cd $CloudAppDir && .venv/bin/python run.py apply-speakers /tmp/zhijiang_speakers.json.gz && rm -f /tmp/zhijiang_speakers.json.gz"
        if ($LASTEXITCODE -ne 0) { throw "云端应用声纹补丁失败。" }
        $SpkApply = ($SpkApplyOut -join "`n") | ConvertFrom-Json
        Write-Host ("    [OK] 云端声纹标注热写入完成：{0} 个分P / {1:N0} 条发言人标签。" -f `
            $SpkApply.parts_applied, $SpkApply.lines_updated) -ForegroundColor Green
    } else {
        Write-Host "    [OK] 声纹发言人标注已与云端 100% 一致。" -ForegroundColor Green
    }
}

Remove-Item -Force -ErrorAction SilentlyContinue $ManifestFile, $DeltaFile, $SpeakersFile

# ── 第 4 步：自动验收公网首页实时统计数据 ──
Write-Host "`n========================================================" -ForegroundColor Green
Write-Host " 增量热同步成功！本次写入 $($ApplyInfo.parts_applied) 个分P / $($ApplyInfo.subtitles_inserted) 条字幕" -ForegroundColor Green
try {
    $AfterStats = Invoke-RestMethod -Uri "$PublicUrl/api/v1/stats" -Headers @{ "Cache-Control" = "no-cache" } -TimeoutSec 10
    $deltaSubs = if ($BeforeStats) { $AfterStats.subtitles_total - $BeforeStats.subtitles_total } else { $ApplyInfo.subtitles_inserted }
    $deltaParts = if ($BeforeStats) { $AfterStats.video_parts_ok - $BeforeStats.video_parts_ok } else { $ApplyInfo.parts_applied }
    Write-Host (" 公网首页实时数据 ($PublicUrl)：") -ForegroundColor Green
    Write-Host ("   已收录纠错字幕 : {0:N0} 条（本次 +{1:N0}）" -f $AfterStats.subtitles_total, $deltaSubs) -ForegroundColor White
    Write-Host ("   覆盖有效分P    : {0:N0} 个分P（本次 +{1:N0}）" -f $AfterStats.video_parts_ok, $deltaParts) -ForegroundColor White
    Write-Host ("   收录时间跨度   : {0} ~ {1}" -f $AfterStats.date_min, $AfterStats.date_max) -ForegroundColor White
    Write-Host ("   库最近更新时间 : {0} (UTC)" -f $AfterStats.latest_updated_at) -ForegroundColor White
} catch {
    Write-Host " 云端库统计已更新：$($ApplyInfo.live_stats | ConvertTo-Json -Compress)" -ForegroundColor White
}
Write-Host "========================================================" -ForegroundColor Green
