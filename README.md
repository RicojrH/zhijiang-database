# 枝江数据库 — 枝江直播回放 AI 字幕语料库与检索系统

面向二创剪辑师与社区粉丝的直播回放字幕语料库，覆盖 **嘉然、贝拉、乃琳、向晚、心宜、思诺** 等成员近 6 年直播回放（通过官方回放 + 第三方归档补齐 `2020-12` 至今持续更新，现已收录 **3,900+ 回放分 P、645 万+ 条纠错字幕**）。系统集成了 B 站官方 AI 字幕增量抓取、早期录播与二创计画缺失场次补全（智能优先无弹幕纯净版并剔除防撞车末尾分 P）、合集级场景标签校准、本地 GPU `faster-whisper` 词级时间戳补录、跨时期双模型声纹发言人识别、专有名词纠错、同音/别称模糊检索与 SQLite WAL 零停服云端增量热同步。

- **作者**：B 站 [`@没品人类之一`](https://space.bilibili.com/396896138)
- **GitHub 仓库**：https://github.com/RicojrH/zhijiang-database
- **在线访问数据库**：https://zjdb.mprlzy.top
- **云端部署指南**：见 [DEPLOY.md](DEPLOY.md)
- **开源协议**：[MIT License](LICENSE)

> **关于数据库文件**：完整的近 6 年 645 万+ 条 SQLite 语料库约 2.47 GB，超过 GitHub 单文件限制，因此仓库仅开源完整源码、建库 Schema、纠错词表、部署/同步脚本与前后端界面；你可按下方步骤自行抓取构建，或直接访问作者部署的公开检索站。

---

## 核心特性

### 1. 公开检索前端与阅读体验（`frontend/`）
- **响应式纯静态前端**：零构建依赖，适配手机 / 平板 / 电脑，支持 **跟随系统 / 浅色 / 深色模式** 切换。
- **多维精准筛选**：按 6 位成员官方应援色区分角色，支持 **直播间多选、发言人多选（声纹识别，单播默认本人，可与直播间自由组合，自动过滤电影/动画/游戏原声）、场景多选（单播/双播/团播/特别场次）、日期范围、BV 号限定与同音/别称模糊扩展**。
- **面向剪辑的检索交互**：支持上下双翻页/跳页、前后 25 秒上下文展开、一键复制台词与精确秒级空降 B 站原视频。
- **实时动态与轻量统计**：首页实时同步收录规模、时间跨度与最新收录回放直达，内置 **站点公告与更新动态看板**（实时读取 `config/announcements.json`）及隐私友好访客统计（`site_metrics.db` 独立隔离、单向截断哈希不存原始 IP）。

### 2. 前后端彻底分离与本地运维控制台（`backend/`）
- **公网只读模式（`public_app`）**：云端仅暴露纯只读 `/api/v1/*` 接口，无写操作、无 Cookie 暴露、不挂载 `/debug`。
- **本地运维台（`local_app` + `/debug`）**：仅本地开发模式启用，集成四大功能面板：
  - 字幕检索验证与已入库场次分页管理（支持场景标签批量修正与人工锁定标识）；
  - B 站扫码登录、「直播回放」合集增量抓取，以及 **二创计画合集补漏 / 标签校准 & 飞书早期录播补全** 一键按钮（支持 Dry-Run 预览与安全停止）；
  - **一键云端增量同步中心**与 **站点公告在线编辑发布**；
  - **本地 GPU ASR 补录控制台**与 **声纹发言人标注控制台**（支持多进程并发、进程树一键停止，并自动检测外部命令行运行的声纹/ASR 任务、每 5 秒同步实时进度与入库明细）。

### 3. 高性能中文全文检索
- 采用 **SQLite WAL + FTS5 外部内容索引 + 中英混排字符级分词**，600 万+ 条语料实测查询响应 **140~190 ms**，首页统计基于 `videos` 元数据表耗时 **< 5 ms**。
- 内置 [config/correction_rules.json](config/correction_rules.json) 分级专有名词纠错规则（支持历史时间界限条件规则与复合词保护）及同音/别称模糊扩展词表（如搜 `小海诺` 自动兼容 `小海螺/小螺`，搜 `心宜` 兼容未替换的 `心仪`），全局繁转简（`zhconv`）。
- 内置轻量规则场景初筛（含 `【突击】` 单播优先策略与人工锁定防覆盖保护）。

### 4. 多源字幕采集、第三方归档补漏与高精度 GPU ASR 补录
- **官方 AI 字幕增量抓取**：精准直连 6 位成员 B 站「直播回放」合集/列表（`series_id`），断点续抓并自动跳过已确认无声/无字幕分 P。
- **第三方早期录播补全与二创计画对齐（`archive_sync.py`）**：
  - 补齐 `2020-12 ~ 2021-08-18` 官方未建回放前的早期录播，以及因超管警告/掐断等原因未生成官方回放的缺失场次；
  - **多 P 智能净化**：同 BV 下同时存在无弹幕/纯净版与弹幕版时优先取无弹幕版（无纯净版时再回退弹幕版），自动剔除防撞车末尾分 P（`防撞`/`❤`/超短占位）与重复弹幕独立 BV；
  - **合集级标签校准**：对照「二创计画」10 个官方分类合集自动修正并锁定存量错误 `scene_type`（同时保留 2025+ 生日会与多人联机突击特例）。
- **本地 GPU 高精度语音补录（`asr-v2`）**：针对无自带字幕分 P，仅拉取纯音频 DASH 流并调用 NVIDIA GPU `faster-whisper (large-v3-turbo)` 多进程分片转写；开启 **词级时间戳对齐、`chunk_length=10s` 与多级智能断句**（单行平均约 2.9 秒，告别超长不断句），提供旧版 ASR 重断句（`resplit-asr`）与高精度重录工具，转完即删临时音频、零视频囤积。

### 5. 跨时期双模型声纹发言人识别引擎
- **双模型 448 维融合**：结合 **CAM++ (192d)** 与 **WeSpeaker CN-Celeb (256d)**，基于 2021~2026 早/中/近期分层单播特征训练判别分类头，克服跨年声线变化与 3D 动捕麦 / 2D 桌面麦域差异。
- **场景先验与句内滑窗**：按发布日期与场次类型动态收窄在籍候选成员（单播观影/反应回启用原声拒识线），结合 0.95s 句内滑窗与局部时间轴平滑，自动识别「单人」「A+B 合唱/换人」与「其他/原声」。
- **低显存流式 GPU 批处理**：优先使用 `onnxruntime-gpu` + Kaldi Fbank（无 CUDA 时自动回退 CPU `sherpa-onnx`），通过流式 WAV 切片、帧预算分桶批处理与多线程预取消除内存/显存 OOM，支持断点续跑。

### 6. SQLite WAL 零停服增量热同步上云
- 本地与云端通过 `sync-manifest` 比对分 P 签名，仅打包传输新增或变动分 P（通常几百 KB ~ 几 MB），云端 `apply-delta` 在 WAL 模式下热写入并自动增量更新 FTS5，**零停服、零全量覆盖**。
- 额外支持 **纯场景标签热补丁（`apply-scene-tags`）**、**纯声纹发言人热补丁（`apply-speakers`，零 FTS5 重写）**、**仅发布代码与公告（`deploy_code_only.ps1`）** 以及 **Nginx 访问日志幂等回填历史 UV/PV（`backfill-metrics`）**。

---

## 目录结构

```text
├── backend/                          # FastAPI 独立后端（公网只读 public_app / 本地运维 local_app）
│   ├── app.py
│   ├── debug_console.html            # 本地一站式运维控制台（仅本地模式挂载在 /debug）
│   └── routers/
│       ├── public.py                 # 公网只读 API (/api/v1/*)
│       └── admin.py                  # 本地私有运维 API (/api/v1/admin/*)
├── frontend/                         # 纯静态公开前端（可直接托管至 Cloudflare Pages / Nginx / FastAPI）
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── config.js                     # 唯一前端配置入口（API_BASE / 作者 / GitHub 链接）
├── config/
│   ├── up_list.json                  # 6 位成员 UID、应援色、粉丝名与「直播回放」合集 ID
│   ├── correction_rules.json         # 分级专有名词纠错规则（含时间条件）与同音模糊词表
│   └── announcements.json            # 站点公告与更新动态（实时生效，无需重启服务）
├── zhijiang_corpus/                  # 核心引擎层
│   ├── db.py / schema.sql            # SQLite WAL + FTS5 建库与表结构定义
│   ├── bilibili_client.py / wbi.py   # B 站 WBI 签名与回放/字幕客户端
│   ├── ingest.py                     # 合集增量抓取与本地 SRT/BCC 导入
│   ├── archive_sync.py               # 飞书早期录播与二创计画 10 合集去重补录、多 P 净化与标签校准
│   ├── cleaner.py / normalizer.py    # 字幕清洗、繁转简与专有名词纠错
│   ├── tokenizer.py / search.py      # 中英混排字符级分词与全文/声纹复合检索
│   ├── asr_backfill.py               # GPU faster-whisper 词级时间戳补录与重断句
│   ├── voiceprint.py                 # CAM++ + WeSpeaker 跨时期声纹发言人识别引擎
│   ├── scene_tagger.py               # 直播场景轻量规则初筛（单播/双播/团播/特别场）
│   ├── sync_delta.py / cloud_ops.py  # SQLite WAL 增量热同步与云端运维编排
│   ├── metrics.py                    # 隐私友好 UV/PV 统计与 Nginx 日志回填
│   └── cli.py                        # 统一命令行入口
├── scripts/
│   ├── deploy_cloud.sh               # Ubuntu 云端公网只读服务一键部署脚本
│   ├── sync_to_cloud.ps1             # Windows 本地增量抓取 + 增量热同步上云脚本
│   ├── deploy_code_only.ps1          # 仅发布线上代码与公告（可选附带增量补丁）
│   ├── run_asr_backfill.ps1          # Windows GPU 多进程分片 ASR 补录脚本
│   ├── retranscribe_legacy_asr.ps1   # 旧版粗断句 ASR 高精度词级时间戳重录脚本
│   ├── sync_archives.py              # 飞书早期录播补全 + 二创计画补漏与标签校准 CLI
│   └── run_vp_then_asr.py            # 2 进程声纹标注 → 2 进程 GPU ASR 自动串联编排脚本
├── tests/
│   ├── test_pipeline.py              # 端到端单元测试
│   └── test_archive_sync.py          # 多 P 净化筛选与归档对齐单元测试
├── DEPLOY.md                         # GitHub 开源与云端公网部署完整教程
├── requirements.txt                  # 核心与云端后端轻量依赖
├── requirements-asr.txt              # 本地 GPU Whisper 补录可选依赖
└── run.py                            # CLI 启动入口
```

---

## 快速开始（本地开发与自建库）

### 1. 基础环境与数据抓取

```powershell
# 1. 创建虚拟环境并安装核心依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 初始化数据库 & 扫码登录 B 站（抓取 ai-zh 官方 AI 字幕需要登录态）
python run.py init-db
python run.py login

# 3. 抓取直播回放 AI 字幕（默认直连 6 人各自的「直播回放」列表，--limit 0 为全量）
python run.py ingest-up all --limit 5

# 4. 专有名词纠错 + 场景初筛 + FTS5 索引优化
python run.py normalize-corpus
python run.py tag-scenes --only-unknown
python run.py optimize-db

# 5. 启动本地服务
#    公开前端:   http://127.0.0.1:8000/
#    本地运维台: http://127.0.0.1:8000/debug
#    API 文档:   http://127.0.0.1:8000/docs
python run.py serve
```

### 2. 进阶功能（GPU ASR 补录 / 声纹标注 / 增量上云）

```powershell
# [可选] GPU 高精度词级时间戳补录（需先安装 requirements-asr.txt 与 FFmpeg）
python run.py backfill-asr --limit 0

# [可选] 对存量旧版粗断句 ASR 执行快速重断句或高精度重录
python run.py resplit-asr
.\scripts\retranscribe_legacy_asr.ps1 -Processes 2 -BatchSize 8

# [可选] 二创计画 10 合集补漏 + 标签校准 & 早期录播补全（支持 --dry-run 预览）
python scripts/sync_archives.py --refresh-erchuang --dry-run
python scripts/sync_archives.py --refresh-erchuang

# [可选] 跨时期声纹发言人标注（优先 CUDA GPU 批量推理，支持断点续跑）
python run.py build-voiceprint-bank
python run.py run-voiceprint --scope multi_and_watch --limit 0

# [可选] Windows 一键增量更新并热同步上云（零停服，首次运行自动引导配置 .cloud_sync.env）
.\scripts\sync_to_cloud.ps1
# 仅同步线上代码与公告（不传数据库）
.\scripts\deploy_code_only.ps1
```

> 本地所有运维功能（扫码登录、增量抓取、场景批量修正、公告编辑、GPU ASR、声纹标注、一键上云与历史访客回填）均可在 `http://127.0.0.1:8000/debug` 中图形化一键操作。

---

## 公网只读 API 概览

云端部署通过 `uvicorn backend.app:public_app`（或 `python run.py serve --public-only`）仅暴露以下只读接口：

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | 服务健康检查 |
| `GET` | `/api/v1/members` | 6 位成员应援色、粉丝名与场景类型元数据 |
| `GET` | `/api/v1/announcements` | 站点公告与更新动态（实时读取，无缓存） |
| `GET` | `/api/v1/stats` | 语料库实时规模、最新回放、成员分布与访客统计（UV/PV） |
| `GET` | `/api/v1/search` | 全文检索（支持 `up_name` 直播间、`speaker` 声纹发言人、场景、日期、BV、模糊扩展与上下文） |
| `GET` | `/api/v1/videos` | 分页浏览已收录的直播回放分 P 列表 |
| `GET` | `/api/v1/videos/{bvid}/subtitles` | 按时间窗口读取指定场次的前后上下文字幕 |

完整部署流程（Ubuntu 一键脚本 `scripts/deploy_cloud.sh`、Caddy/Nginx 反代与 Cloudflare Pages 托管）请参考 [DEPLOY.md](DEPLOY.md)。

---

## 致谢与反馈

欢迎通过 B 站 [`@没品人类之一`](https://space.bilibili.com/396896138) 或 [GitHub Issues](https://github.com/RicojrH/zhijiang-database/issues) 反馈错词、场景标签误差、Bug 与功能建议！
