param(
    [string]$DeltaFile = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ConfFile = Join-Path $Root ".cloud_sync.env"
$Conf = @{
    CLOUD_HOST    = ""
    CLOUD_APP_DIR = "/opt/zhijiang"
    PUBLIC_URL    = "https://zjdb.mprlzy.top"
}
if (Test-Path $ConfFile) {
    Get-Content $ConfFile -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)\s*$') {
            $Conf[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
}
if (-not $Conf["CLOUD_HOST"]) {
    throw "未找到 .cloud_sync.env 中的 CLOUD_HOST 配置，请先运行一次 一键更新并同步上云.bat。"
}

$CloudHost   = $Conf["CLOUD_HOST"]
$CloudAppDir = $Conf["CLOUD_APP_DIR"]
$PublicUrl   = $Conf["PUBLIC_URL"].TrimEnd("/")
$AnnFile     = Join-Path $Root "config\announcements.json"
if (-not (Test-Path $AnnFile)) {
    throw "未找到公告配置文件: $AnnFile"
}

# 校验本地 JSON 格式合法
$null = Get-Content $AnnFile -Raw -Encoding UTF8 | ConvertFrom-Json
$AnnB64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($AnnFile))

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " 枝江数据库 · 仅发布线上代码与公告（不上传本地数据库）" -ForegroundColor Cyan
Write-Host " 目标服务器: $CloudHost ($CloudAppDir)" -ForegroundColor Gray
Write-Host " 公开站点:   $PublicUrl" -ForegroundColor Gray
Write-Host "========================================================`n" -ForegroundColor Cyan

$ApplyDeltaBlock = ""
if ($DeltaFile) {
    $ResolvedDelta = if ([System.IO.Path]::IsPathRooted($DeltaFile)) { $DeltaFile } else { Join-Path $Root $DeltaFile }
    if (-not (Test-Path $ResolvedDelta)) {
        throw "未找到指定的增量补丁文件: $ResolvedDelta"
    }
    $DeltaKb = [math]::Round((Get-Item $ResolvedDelta).Length / 1KB, 1)
    Write-Host "==> 正在上传指定增量补丁 ($DeltaKb KB) 到 $CloudHost ..." -ForegroundColor Cyan
    & scp $ResolvedDelta "${CloudHost}:/tmp/zhijiang_quick_delta.json.gz"
    if ($LASTEXITCODE -ne 0) {
        throw "SCP 上传补丁失败，请检查 SSH 连接或密码。"
    }
    $ApplyDeltaBlock = @"
.venv/bin/python run.py apply-delta /tmp/zhijiang_quick_delta.json.gz
rm -f /tmp/zhijiang_quick_delta.json.gz
"@
}

$RemoteCmd = @"
set -e
cd $CloudAppDir
git checkout -- config/announcements.json 2>/dev/null || true
git pull --ff-only origin main
mkdir -p config
printf '%s' '$AnnB64' | base64 -d > config/announcements.json
$ApplyDeltaBlock
SVC=`$(systemctl list-unit-files --type=service --no-legend | awk '{print `$1}' | grep -iE 'zhijiang|zjdb' | head -n 1 || true)
if [ -n "`$SVC" ]; then
  echo "[remote] restarting `$SVC ..."
  systemctl restart "`$SVC"
  sleep 1
  systemctl is-active "`$SVC"
else
  echo "[remote] 未找到 systemd 服务，请检查 uvicorn 进程"
fi
"@
$RemoteCmd = $RemoteCmd -replace "`r`n", "`n"

Write-Host "==> 正在连接服务器拉取最新代码、同步公告$(if ($DeltaFile) { '、热写入指定补丁' })并平滑重启服务..." -ForegroundColor Cyan
& ssh $CloudHost $RemoteCmd
if ($LASTEXITCODE -ne 0) {
    throw "远程发布失败，请检查 SSH 连接或密码。"
}

Write-Host "`n==> 正在验收公网实时接口 ($PublicUrl/api/v1/stats)..." -ForegroundColor Cyan
Start-Sleep -Seconds 1
try {
    $rawBytes = (Invoke-WebRequest -Uri "$PublicUrl/api/v1/stats" -Headers @{ "Cache-Control" = "no-cache" } -TimeoutSec 12 -UseBasicParsing).RawContentStream.ToArray()
    $st = [System.Text.Encoding]::UTF8.GetString($rawBytes) | ConvertFrom-Json
    Write-Host "  已收录字幕   : $($st.subtitles_total) 条 | 有效分P: $($st.video_parts_ok)" -ForegroundColor White
    Write-Host "  收录时间跨度 : $($st.date_min) ~ $($st.date_max)" -ForegroundColor White
    if ($st.latest_video) {
        Write-Host "  最新收录录播 : [$($st.latest_video.up_name)] $($st.latest_video.title)" -ForegroundColor White
    }
    $annCount = if ($st.announcements) { $st.announcements.Count } else { 0 }
    Write-Host "  当前生效公告 : $annCount 条" -ForegroundColor Green
    if ($st.announcements) {
        foreach ($a in $st.announcements) {
            Write-Host "    - [$($a.tag)] $($a.content) ($($a.date))" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "[提示] 线上服务已重启，请直接刷新 $PublicUrl 确认显示效果。" -ForegroundColor Yellow
}

if ($DeltaFile) {
    Write-Host "`n[OK] 线上代码、公告与指定增量补丁发布完成！" -ForegroundColor Green
} else {
    Write-Host "`n[OK] 线上代码与公告发布完成（云端数据库未做任何修改）。" -ForegroundColor Green
}
