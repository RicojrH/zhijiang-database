(function () {
  "use strict";

  const CFG = window.ZHIJIANG_CONFIG || {};
  const API_BASE = (CFG.API_BASE || "").replace(/\/+$/, "");

  const MEMBER_THEMES_LIGHT = {
    "嘉然": { color: "#E799B0", bg: "#FDF3F6", ink: "#9C3A5A", fan: "嘉心糖", official: "嘉然今天吃什么" },
    "贝拉": { color: "#DB7D74", bg: "#FDF3F2", ink: "#A13A31", fan: "贝极星", official: "贝拉kira" },
    "乃琳": { color: "#576690", bg: "#F1F3F8", ink: "#364369", fan: "奶淇琳", official: "乃琳Queen" },
    "向晚": { color: "#9AC8E2", bg: "#F0F8FC", ink: "#225F80", fan: "顶碗人", official: "向晚大魔王" },
    "心宜": { color: "#C93773", bg: "#FCEFF4", ink: "#9D2256", fan: "心球仪", official: "心宜不是心仪" },
    "思诺": { color: "#7252C0", bg: "#F4F1FB", ink: "#513399", fan: "小海诺", official: "思诺snow" }
  };

  const MEMBER_THEMES_DARK = {
    "嘉然": { color: "#E799B0", bg: "rgba(231, 153, 176, 0.15)", ink: "#F4B8C9", fan: "嘉心糖", official: "嘉然今天吃什么" },
    "贝拉": { color: "#DB7D74", bg: "rgba(219, 125, 116, 0.15)", ink: "#F0A39B", fan: "贝极星", official: "贝拉kira" },
    "乃琳": { color: "#8798C8", bg: "rgba(135, 152, 200, 0.16)", ink: "#B3C1E8", fan: "奶淇琳", official: "乃琳Queen" },
    "向晚": { color: "#9AC8E2", bg: "rgba(154, 200, 226, 0.15)", ink: "#B5DCF2", fan: "顶碗人", official: "向晚大魔王" },
    "心宜": { color: "#E05C94", bg: "rgba(224, 92, 148, 0.16)", ink: "#F594BC", fan: "心球仪", official: "心宜不是心仪" },
    "思诺": { color: "#9A7EE0", bg: "rgba(154, 126, 224, 0.16)", ink: "#C3B0F2", fan: "小海诺", official: "思诺snow" }
  };

  const MEMBER_THEMES = MEMBER_THEMES_LIGHT;

  const THEME_ICONS = {
    auto: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`,
    light: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`,
    dark: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`
  };
  const THEME_LABELS = { auto: "跟随系统", light: "浅色", dark: "深色" };
  const darkMediaQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  const SCENE_LABELS = {
    solo: "单播",
    duo: "双播",
    group: "团播",
    special: "特别场次",
    unknown: "未标注"
  };

  const state = {
    selectedUps: new Set(),
    selectedScenes: new Set(),
    currentPage: 1,
    totalPages: 0,
    totalHits: 0,
    pageLimit: 20,
    hasSearched: false,
    announcementsExpanded: false,
    members: [],
    stats: null,
    site: {
      author_name: CFG.AUTHOR_NAME || "没品人类之一",
      author_bilibili_url: CFG.AUTHOR_BILIBILI_URL || "https://space.bilibili.com/396896138",
      github_url: CFG.GITHUB_REPO_URL || ""
    }
  };

  const $ = (id) => document.getElementById(id);

  function esc(str) {
    return String(str || "").replace(/[&<>"]/g, (ch) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[ch]
    ));
  }

  function formatNumber(n) {
    return Number(n || 0).toLocaleString("zh-CN");
  }

  function isDarkMode() {
    return document.documentElement.dataset.theme === "dark";
  }

  function getSavedThemeMode() {
    try {
      const m = localStorage.getItem("zj_theme") || "auto";
      return (m === "light" || m === "dark" || m === "auto") ? m : "auto";
    } catch (e) {
      return "auto";
    }
  }

  function updateThemeButton(mode) {
    const iconEl = $("themeIcon");
    const labelEl = $("themeLabel");
    const btn = $("themeToggleBtn");
    if (iconEl) iconEl.innerHTML = THEME_ICONS[mode] || THEME_ICONS.auto;
    if (labelEl) labelEl.textContent = THEME_LABELS[mode] || "跟随系统";
    if (btn) {
      const curText = isDarkMode() ? "当前深色" : "当前浅色";
      btn.title = `主题模式：${THEME_LABELS[mode] || "跟随系统"}（${curText}），点击切换`;
    }
  }

  function syncVisibleHitCardThemes() {
    document.querySelectorAll(".hit-card[data-up]").forEach((card) => {
      const t = getTheme(card.dataset.up || "");
      card.style.setProperty("--m-color", t.color);
      card.style.setProperty("--m-bg", t.bg);
      card.style.setProperty("--m-ink", t.ink);
    });
  }

  function applyThemeMode(mode, persist) {
    const safeMode = (mode === "light" || mode === "dark" || mode === "auto") ? mode : "auto";
    if (persist) {
      try { localStorage.setItem("zj_theme", safeMode); } catch (e) {}
    }
    const dark = safeMode === "dark" || (safeMode === "auto" && Boolean(darkMediaQuery && darkMediaQuery.matches));
    document.documentElement.dataset.themeMode = safeMode;
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    updateThemeButton(safeMode);
    applyThemeAccent();
    renderMemberChips();
    renderLandingStatsAndCards();
    syncVisibleHitCardThemes();
  }

  function cycleThemeMode() {
    const order = ["auto", "light", "dark"];
    const cur = getSavedThemeMode();
    const next = order[(order.indexOf(cur) + 1) % order.length];
    applyThemeMode(next, true);
  }

  function getTheme(upName) {
    const dark = isDarkMode();
    const map = dark ? MEMBER_THEMES_DARK : MEMBER_THEMES_LIGHT;
    return map[upName] || (dark
      ? { color: "#8798C8", bg: "rgba(135, 152, 200, 0.16)", ink: "#B3C1E8", fan: "枝江", official: upName || "" }
      : { color: "#576690", bg: "#F1F3F8", ink: "#364369", fan: "枝江", official: upName || "" }
    );
  }

  function applyThemeAccent() {
    const root = document.documentElement;
    const activeUp = state.selectedUps.size === 1 ? Array.from(state.selectedUps)[0] : "";
    const t = getTheme(activeUp);
    root.style.setProperty("--accent", t.color);
    root.style.setProperty("--accent-bg", t.bg);
    root.style.setProperty("--accent-ink", t.ink);
    updateSearchPlaceholder();
  }

  function updateSearchPlaceholder() {
    const input = $("qInput");
    if (!input) return;
    const ups = Array.from(state.selectedUps);
    if (ups.length === 1) {
      const upName = ups[0];
      const upStat = state.stats && state.stats.by_up
        ? state.stats.by_up.find((u) => u.up_name === upName)
        : null;
      const countText = upStat ? ` ${Math.round(upStat.subtitle_lines / 10000)} 万+ 条` : "";
      input.placeholder = `在 ${upName} 的${countText}字幕中搜索台词、热梗或名场面…（空格分隔多词）`;
    } else if (ups.length > 1) {
      input.placeholder = `在 ${ups.join("、")} 的字幕中搜索台词、热梗或名场面…（空格分隔多词）`;
    } else {
      input.placeholder = "搜台词、搜热梗、搜名场面… 如：小作文 / 嘉心糖 / 心球仪 / 黑魔法学院（空格分隔多词）";
    }
  }

  function highlightText(text, q, expandedMap) {
    let out = esc(text);
    const terms = new Set((q || "").trim().split(/\s+/).filter(Boolean));
    if (expandedMap) {
      Object.values(expandedMap).forEach((arr) => {
        (arr || []).forEach((v) => terms.add(v));
      });
    }
    const sorted = Array.from(terms).sort((a, b) => b.length - a.length);
    for (const t of sorted) {
      const safe = t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (!safe) continue;
      out = out.replace(new RegExp(safe, "gi"), (m) => `<mark>${m}</mark>`);
    }
    return out;
  }

  function hasActiveQueryCondition() {
    const q = $("qInput").value.trim();
    const bvid = $("bvidFilter").value.trim();
    const df = $("dateFrom").value;
    const dt = $("dateTo").value;
    return Boolean(q || bvid || df || dt);
  }

  function showLandingMode() {
    state.hasSearched = false;
    $("landingBoard").classList.remove("hidden");
    $("resultsSection").classList.add("hidden");
  }

  function showResultsMode() {
    state.hasSearched = true;
    $("landingBoard").classList.add("hidden");
    $("resultsSection").classList.remove("hidden");
  }

  function renderMemberChips() {
    const wrap = $("memberStrip");
    const members = state.members.length
      ? state.members
      : Object.keys(MEMBER_THEMES).map((up_name) => ({ up_name, ...MEMBER_THEMES[up_name] }));

    const allActive = state.selectedUps.size === 0 ? "active" : "";
    const allChipStyle = isDarkMode()
      ? "--chip-color:#8798C8;--chip-bg:rgba(135,152,200,0.16);--chip-ink:#EEF1F8;"
      : "--chip-color:#576690;--chip-bg:#F1F3F8;--chip-ink:#1B2232;";
    const allBtn = `
      <span class="strip-label">成员（可多选）</span>
      <button type="button" class="member-chip ${allActive}"
        data-up="" style="${allChipStyle}">
        全部成员
      </button>
    `;
    const memberBtns = members.map((m) => {
      const t = getTheme(m.up_name);
      const color = m.color || t.color;
      const active = state.selectedUps.has(m.up_name) ? "active" : "";
      return `
        <button type="button" class="member-chip ${active}" data-up="${esc(m.up_name)}"
          style="--chip-color:${color};--chip-bg:${t.bg};--chip-ink:${t.ink};">
          <span class="color-dot" style="background:${color};"></span>
          ${esc(m.up_name)}
        </button>
      `;
    }).join("");

    wrap.innerHTML = allBtn + memberBtns;
    wrap.querySelectorAll(".member-chip").forEach((btn) => {
      btn.addEventListener("click", () => toggleMember(btn.dataset.up || "", false));
    });
  }

  function renderSceneChips() {
    const wrap = $("sceneStrip");
    if (!wrap) return;
    wrap.querySelectorAll(".scene-chip").forEach((btn) => {
      const sc = btn.dataset.scene || "";
      const isActive = sc === "" ? state.selectedScenes.size === 0 : state.selectedScenes.has(sc);
      btn.classList.toggle("active", isActive);
    });
  }

  function toggleMember(upName, focusInput, exclusive) {
    if (!upName) {
      state.selectedUps.clear();
    } else if (exclusive) {
      state.selectedUps.clear();
      state.selectedUps.add(upName);
    } else {
      if (state.selectedUps.has(upName)) {
        state.selectedUps.delete(upName);
      } else {
        state.selectedUps.add(upName);
      }
      const totalMembers = state.members.length || 6;
      if (state.selectedUps.size >= totalMembers) {
        state.selectedUps.clear();
      }
    }
    applyThemeAccent();
    renderMemberChips();
    if (hasActiveQueryCondition()) {
      runSearch(1, false);
    } else {
      showLandingMode();
      if (focusInput) {
        $("qInput").focus();
      }
    }
  }

  function toggleScene(sceneKey) {
    if (!sceneKey) {
      state.selectedScenes.clear();
    } else if (state.selectedScenes.has(sceneKey)) {
      state.selectedScenes.delete(sceneKey);
    } else {
      state.selectedScenes.add(sceneKey);
    }
    renderSceneChips();
    if (hasActiveQueryCondition()) {
      runSearch(1, false);
    }
  }

  function formatLocalDateTime(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    if (Number.isNaN(d.getTime())) return String(isoStr);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function renderAnnouncements(items) {
    const card = $("announcementCard");
    const listEl = $("announcementList");
    const metaEl = $("announcementMeta");
    const toggleBtn = $("announcementToggle");
    if (!card || !listEl) return;

    const list = Array.isArray(items) ? items : [];
    if (!list.length) {
      card.classList.add("hidden");
      listEl.innerHTML = "";
      return;
    }

    card.classList.remove("hidden");
    const previewCount = 3;
    const expanded = state.announcementsExpanded;
    const visible = expanded ? list : list.slice(0, previewCount);

    if (metaEl) {
      metaEl.textContent = list.length > 1 ? `共 ${list.length} 条动态` : (list[0].date || "");
    }

    listEl.innerHTML = visible.map((it) => {
      const level = ["info", "change", "warn"].includes(it.level) ? it.level : "info";
      const tag = esc(it.tag || "公告");
      const dateHtml = it.date ? `<span class="announcement-date">${esc(it.date)}</span>` : "";
      return `
        <li class="announcement-item">
          <span class="announcement-tag tag-${level}">${tag}</span>
          <span class="announcement-text">${esc(it.content)}</span>
          ${dateHtml}
        </li>
      `;
    }).join("");

    if (toggleBtn) {
      if (list.length > previewCount) {
        toggleBtn.classList.remove("hidden");
        toggleBtn.textContent = expanded ? "收起较早公告" : `展开全部 ${list.length} 条公告`;
      } else {
        toggleBtn.classList.add("hidden");
      }
    }
  }

  function renderVisitorMetrics(v) {
    const el = $("footerVisitorStats");
    if (!el) return;
    if (!v || (Number(v.total_uv || 0) <= 0 && Number(v.total_pv || 0) <= 0)) {
      el.classList.add("hidden");
      return;
    }
    el.classList.remove("hidden");
    el.innerHTML =
      `<span class="footer-metrics-dot" aria-hidden="true"></span>` +
      `<span>累计访客 <strong>${formatNumber(v.total_uv)}</strong> 人（今日 <strong>${formatNumber(v.today_uv)}</strong>）· 累计访问 <strong>${formatNumber(v.total_pv)}</strong> 次</span>`;
  }

  function renderLandingStatsAndCards() {
    const st = state.stats;
    if (st) {
      $("statTotalLines").textContent = formatNumber(st.subtitles_total);
      $("statTotalParts").textContent = `${formatNumber(st.video_parts_ok)} 个分P`;
      $("statDateRange").textContent = st.date_min && st.date_max
        ? `${st.date_min} ~ ${st.date_max}`
        : "—";

      const latestEl = $("statLatestVideo");
      if (latestEl) {
        const lv = st.latest_video;
        if (lv && lv.bvid) {
          const t = getTheme(lv.up_name);
          const cleanTitle = (lv.title || lv.bvid).replace(/^【直播回放】\s*/, "");
          const partSuffix = lv.page && lv.page > 1 ? ` · P${lv.page}` : "";
          const fullTitle = `${lv.up_name ? `[${lv.up_name}] ` : ""}${lv.title || lv.bvid}${partSuffix}`;
          const upHtml = lv.up_name
            ? `<span class="stat-latest-up" style="--latest-ink:${t.ink};">[${esc(lv.up_name)}]</span>`
            : "";
          const href = esc(lv.video_url || `https://www.bilibili.com/video/${lv.bvid}/`);
          latestEl.innerHTML =
            `<span class="stat-latest-label">最新录播：</span>` +
            `<a href="${href}" target="_blank" rel="noopener noreferrer" title="${esc(fullTitle)}">` +
            `${upHtml}${esc(cleanTitle)}${esc(partSuffix)}</a>`;
        } else {
          latestEl.innerHTML = "";
        }
      }

      const updatedEl = $("statUpdatedAt");
      if (updatedEl) {
        const ts = formatLocalDateTime(st.latest_updated_at);
        updatedEl.textContent = ts ? `数据库同步时间：${ts}` : "";
      }

      renderAnnouncements(st.announcements);
      renderVisitorMetrics(st.visitors);
    }

    const byUpMap = {};
    if (st && st.by_up) {
      st.by_up.forEach((row) => { byUpMap[row.up_name] = row; });
    }

    const members = state.members.length
      ? state.members
      : Object.keys(MEMBER_THEMES).map((up_name) => ({ up_name, ...MEMBER_THEMES[up_name] }));

    $("memberGrid").innerHTML = members.map((m) => {
      const t = getTheme(m.up_name);
      const color = m.color || t.color;
      const stat = byUpMap[m.up_name];
      const partsText = stat ? `${formatNumber(stat.ok_parts)} 个分P` : "—";
      const linesText = stat ? `${formatNumber(stat.subtitle_lines)} 条字幕` : "—";
      return `
        <button type="button" class="member-card" data-up="${esc(m.up_name)}"
          style="--m-color:${color};--m-bg:${t.bg};--m-ink:${t.ink};">
          <div class="member-card-top">
            <div class="member-id">
              <span class="member-avatar">${esc(m.up_name.slice(0, 1))}</span>
              <div style="min-width:0;">
                <div class="member-name">${esc(m.up_name)}</div>
                <div class="member-official">${esc(m.official_name || t.official)}</div>
              </div>
            </div>
            <span class="fan-pill">${esc(m.fan_name || t.fan)}</span>
          </div>
          <div class="member-card-stats">
            <span>${partsText}</span>
            <span>${linesText}</span>
          </div>
        </button>
      `;
    }).join("");

    $("memberGrid").querySelectorAll(".member-card").forEach((card) => {
      card.addEventListener("click", () => {
        toggleMember(card.dataset.up || "", true, true);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    });

    const hotList = CFG.HOT_KEYWORDS || [];
    $("hotTags").innerHTML = hotList.map((kw) => (
      `<button type="button" class="hot-tag" data-kw="${esc(kw)}">${esc(kw)}</button>`
    )).join("");
    $("hotTags").querySelectorAll(".hot-tag").forEach((tag) => {
      tag.addEventListener("click", () => {
        $("qInput").value = tag.dataset.kw || "";
        syncClearBtn();
        runSearch(1, false);
      });
    });
  }

  function buildPageNumbers(current, total) {
    if (total <= 7) {
      return Array.from({ length: total }, (_, i) => i + 1);
    }
    const pages = [1];
    let left = Math.max(2, current - 2);
    let right = Math.min(total - 1, current + 2);
    if (current <= 3) right = Math.min(total - 1, 5);
    if (current >= total - 2) left = Math.max(2, total - 4);
    if (left > 2) pages.push("...");
    for (let i = left; i <= right; i++) pages.push(i);
    if (right < total - 1) pages.push("...");
    pages.push(total);
    return pages;
  }

  function renderPager(containerId, position) {
    const el = $(containerId);
    if (!el) return;
    if (state.totalPages <= 1) {
      el.classList.add("hidden");
      el.innerHTML = "";
      return;
    }
    el.classList.remove("hidden");
    const cur = state.currentPage;
    const total = state.totalPages;
    const nums = buildPageNumbers(cur, total);

    const numButtons = nums.map((item) => {
      if (item === "...") {
        return `<span class="page-ellipsis">…</span>`;
      }
      const active = item === cur ? "active" : "";
      return `<button type="button" class="page-btn ${active}" data-page="${item}">${item}</button>`;
    }).join("");

    el.innerHTML = `
      <div class="pager-nav">
        <button type="button" class="page-btn" data-page="1" ${cur <= 1 ? "disabled" : ""}>首页</button>
        <button type="button" class="page-btn" data-page="${cur - 1}" ${cur <= 1 ? "disabled" : ""}>上一页</button>
        ${numButtons}
        <button type="button" class="page-btn" data-page="${cur + 1}" ${cur >= total ? "disabled" : ""}>下一页</button>
        <button type="button" class="page-btn" data-page="${total}" ${cur >= total ? "disabled" : ""}>末页</button>
      </div>
      <div class="pager-jump">
        <span>跳至</span>
        <input type="number" min="1" max="${total}" value="${cur}" aria-label="跳转页码" class="jump-input" />
        <span>/ ${formatNumber(total)} 页</span>
        <button type="button" class="page-btn jump-go">跳转</button>
      </div>
    `;

    el.querySelectorAll("button[data-page]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = parseInt(btn.dataset.page, 10);
        if (!Number.isNaN(target) && target >= 1 && target <= total && target !== cur) {
          runSearch(target, position === "bottom");
        }
      });
    });

    const jInput = el.querySelector(".jump-input");
    const jGo = el.querySelector(".jump-go");
    const triggerJump = () => {
      const val = parseInt(jInput.value, 10);
      if (!Number.isNaN(val)) {
        const clamped = Math.max(1, Math.min(total, val));
        runSearch(clamped, position === "bottom");
      }
    };
    jGo.addEventListener("click", triggerJump);
    jInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") triggerJump();
    });
  }

  async function runSearch(page, scrollToResults) {
    if (!hasActiveQueryCondition()) {
      showLandingMode();
      return;
    }

    state.pageLimit = parseInt($("pageSizeFilter").value || "20", 10);
    const targetPage = Math.max(1, page || 1);
    const offset = (targetPage - 1) * state.pageLimit;
    const q = $("qInput").value.trim();

    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (state.selectedUps.size > 0) {
      params.set("up_name", Array.from(state.selectedUps).join(","));
    }
    if (state.selectedScenes.size > 0) {
      params.set("scene_type", Array.from(state.selectedScenes).join(","));
    }
    const df = $("dateFrom").value;
    const dt = $("dateTo").value;
    const bv = $("bvidFilter").value.trim();
    const sort = $("sortFilter").value;
    const fuzzy = $("fuzzyToggle").checked;
    const speakerToggleEl = $("speakerFilterToggle");
    const speakerFilter = Boolean(speakerToggleEl && speakerToggleEl.checked);

    if (df) params.set("date_from", df);
    if (dt) params.set("date_to", dt);
    if (bv) params.set("bvid", bv);
    params.set("sort", sort);
    params.set("fuzzy", fuzzy ? "true" : "false");
    params.set("speaker_filter", speakerFilter ? "true" : "false");
    params.set("limit", String(state.pageLimit));
    params.set("offset", String(offset));

    showResultsMode();
    $("hitList").innerHTML = `<div class="state-box">正在检索语料库…</div>`;
    $("pagerTop").classList.add("hidden");
    $("pagerBottom").classList.add("hidden");

    try {
      const t0 = performance.now();
      const resp = await fetch(`${API_BASE}/api/v1/search?${params.toString()}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const elapsedMs = Math.round(performance.now() - t0);

      const p = data.pagination || {};
      state.totalHits = p.total || 0;
      state.currentPage = p.page || targetPage;
      state.totalPages = p.total_pages || (state.totalHits > 0 ? Math.ceil(state.totalHits / state.pageLimit) : 0);

      const startIdx = state.totalHits === 0 ? 0 : offset + 1;
      const endIdx = offset + (data.items ? data.items.length : 0);
      $("resultSummaryText").textContent =
        `共命中 ${formatNumber(state.totalHits)} 条字幕（第 ${state.currentPage} / ${Math.max(1, state.totalPages)} 页，显示 ${formatNumber(startIdx)}-${formatNumber(endIdx)}，耗时 ${elapsedMs}ms）`;

      const expEl = $("expandedHint");
      const expEntries = Object.entries(data.fuzzy_expanded || {});
      if (expEntries.length) {
        expEl.textContent = "已同音扩展：" + expEntries.map(([k, v]) => `${k} → ${v.join(" / ")}`).join("；");
        expEl.classList.remove("hidden");
      } else {
        expEl.classList.add("hidden");
      }

      if (!data.items || !data.items.length) {
        $("hitList").innerHTML = `
          <div class="state-box">
            没有找到匹配的字幕条目。可以尝试缩短关键词、开启“同音/别称模糊扩展”或切换至“全部成员”。
          </div>
        `;
        return;
      }

      $("hitList").innerHTML = data.items.map((it) => {
        const t = getTheme(it.up_name);
        const sceneName = SCENE_LABELS[it.scene_type] || it.scene_type;
        let speakerBadge = "";
        if (it.speaker_name) {
          const pct = Math.round(Number(it.speaker_conf || 0) * 100);
          if (it.speaker_name === "其他/原声") {
            speakerBadge = `<span class="badge badge-speaker-other" title="声纹判定为非本库 6 位成员人声（如电影/动画/游戏原声或嘉宾）">声纹: 其他/原声</span>`;
          } else if (it.speaker_name === it.up_name) {
            speakerBadge = `<span class="badge badge-speaker-ok" title="声纹确认本人发言（置信度 ${pct}%）">声纹: 本人 ✓</span>`;
          } else {
            const firstSp = String(it.speaker_name).split("+")[0];
            const st = getTheme(firstSp);
            speakerBadge = `<span class="badge badge-speaker" style="--sp-color:${st.color};--sp-bg:${st.bg};--sp-ink:${st.ink};" title="声纹识别实际发言人（置信度 ${pct}%）">发言: ${esc(it.speaker_name)}</span>`;
          }
        }
        return `
          <article class="hit-card" data-up="${esc(it.up_name)}" style="--m-color:${t.color};--m-bg:${t.bg};--m-ink:${t.ink};">
            <div class="hit-meta">
              <span class="badge badge-member">
                <span class="color-dot" style="background:${t.color};"></span>
                ${esc(it.up_name)}
              </span>
              <span class="badge badge-scene">${esc(sceneName)}</span>
              ${speakerBadge}
              <span class="badge badge-time">${esc(it.start_hms)} ~ ${esc(it.end_hms)}</span>
              <span>${esc(it.pub_date)}</span>
              <span>${esc(it.bvid)} · P${it.page}</span>
            </div>
            <div class="hit-content">${highlightText(it.content, q, data.fuzzy_expanded)}</div>
            <div class="hit-foot">
              <div class="hit-title">${esc(it.title)}${it.part_title && it.part_title !== it.title ? " · " + esc(it.part_title) : ""}</div>
              <div class="hit-actions">
                <button type="button" class="act-link ctx-btn"
                  data-id="${it.id}" data-bvid="${esc(it.bvid)}" data-page="${it.page}" data-start="${it.start_sec}">
                  前后上下文
                </button>
                <button type="button" class="act-link copy-btn" data-copy="${esc(it.jump_url)}">
                  复制链接
                </button>
                <a class="act-link" href="${esc(it.jump_url)}" target="_blank" rel="noopener noreferrer"
                  title="打开 B 站原视频，若播放器恢复历史进度可手动拖至 ${esc(it.start_hms)}">
                  B站原片 (${esc(it.start_hms)}) ↗
                </a>
              </div>
            </div>
            <div class="ctx-drawer hidden" id="ctx-${it.id}"></div>
          </article>
        `;
      }).join("");

      $("hitList").querySelectorAll(".ctx-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          toggleContext(
            parseInt(btn.dataset.id, 10),
            btn.dataset.bvid,
            parseInt(btn.dataset.page, 10),
            parseFloat(btn.dataset.start)
          );
        });
      });

      $("hitList").querySelectorAll(".copy-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          navigator.clipboard.writeText(btn.dataset.copy || "").then(() => {
            const orig = btn.textContent;
            btn.textContent = "已复制";
            setTimeout(() => { btn.textContent = orig; }, 1200);
          });
        });
      });

      renderPager("pagerTop", "top");
      renderPager("pagerBottom", "bottom");

      if (scrollToResults) {
        const sec = $("resultsSection");
        const topY = sec.getBoundingClientRect().top + window.scrollY - 76;
        window.scrollTo({ top: Math.max(0, topY), behavior: "smooth" });
      }
    } catch (err) {
      $("hitList").innerHTML = `<div class="state-box">检索请求失败：${esc(err.message)}。请确认后端 API 服务已正常运行。</div>`;
    }
  }

  async function toggleContext(subId, bvid, page, startSec) {
    const box = $(`ctx-${subId}`);
    if (!box) return;
    if (!box.classList.contains("hidden")) {
      box.classList.add("hidden");
      return;
    }
    box.classList.remove("hidden");
    box.textContent = "正在加载前后 25 秒上下文…";
    try {
      const qs = new URLSearchParams({
        page: String(page),
        start_sec: String(Math.max(0, startSec - 25)),
        end_sec: String(startSec + 25),
        limit: "30"
      });
      const r = await fetch(`${API_BASE}/api/v1/videos/${encodeURIComponent(bvid)}/subtitles?${qs.toString()}`);
      const data = await r.json();
      if (!data.items || !data.items.length) {
        box.textContent = "暂无上下文记录。";
        return;
      }
      box.innerHTML = data.items.map((x) => {
        const spHtml = x.speaker_name
          ? `<span class="ctx-speaker">[${esc(x.speaker_name)}]</span>`
          : "";
        return `
        <div class="ctx-row ${x.id === subId ? "is-hit" : ""}">
          <span class="ctx-ts">${Number(x.start_sec).toFixed(1)}s</span>
          ${spHtml}
          <span>${esc(x.content)}</span>
        </div>
      `;
      }).join("");
    } catch (e) {
      box.textContent = "加载上下文失败。";
    }
  }

  function renderAuthorAndGithub() {
    const author = state.site.author_name || "没品人类之一";
    const biliUrl = state.site.author_bilibili_url || "https://space.bilibili.com/396896138";
    const ghUrl = CFG.GITHUB_REPO_URL || state.site.github_url || "";

    $("footerAuthorLink").textContent = `@${author}`;
    $("footerAuthorLink").href = biliUrl;
    $("modalAuthorLink").textContent = `B站 @${author}`;
    $("modalAuthorLink").href = biliUrl;

    const footerGh = $("footerGithubLink");
    const modalGh = $("modalGithubLink");
    if (ghUrl) {
      footerGh.href = ghUrl;
      footerGh.classList.remove("hidden");
      modalGh.href = ghUrl;
      modalGh.textContent = ghUrl;
    } else {
      footerGh.classList.add("hidden");
      modalGh.removeAttribute("href");
      modalGh.textContent = "仓库创建后可在 frontend/config.js 填入 GitHub 链接";
    }
  }

  function syncClearBtn() {
    $("clearInputBtn").classList.toggle("hidden", !$("qInput").value);
  }

  let lastStatsFetchAt = 0;

  async function refreshStats(force) {
    const now = Date.now();
    if (!force && now - lastStatsFetchAt < 30000) return;
    lastStatsFetchAt = now;
    try {
      const sRes = await fetch(`${API_BASE}/api/v1/stats`, { cache: "no-store" });
      if (sRes.ok) {
        state.stats = await sRes.json();
        renderLandingStatsAndCards();
        updateSearchPlaceholder();
      }
    } catch (err) {
      // 忽略后台静默刷新失败
    }
  }

  function resetAll() {
    $("qInput").value = "";
    $("dateFrom").value = "";
    $("dateTo").value = "";
    $("bvidFilter").value = "";
    $("sortFilter").value = "date_desc";
    $("fuzzyToggle").checked = true;
    const spToggle = $("speakerFilterToggle");
    if (spToggle) spToggle.checked = false;
    state.selectedUps.clear();
    state.selectedScenes.clear();
    applyThemeAccent();
    renderMemberChips();
    renderSceneChips();
    syncClearBtn();
    showLandingMode();
    refreshStats(true);
  }

  async function init() {
    applyThemeMode(getSavedThemeMode(), false);
    renderSceneChips();
    renderAuthorAndGithub();

    const themeBtn = $("themeToggleBtn");
    if (themeBtn) {
      themeBtn.addEventListener("click", cycleThemeMode);
    }
    if (darkMediaQuery && darkMediaQuery.addEventListener) {
      darkMediaQuery.addEventListener("change", () => {
        if (getSavedThemeMode() === "auto") {
          applyThemeMode("auto", false);
        }
      });
    }

    $("sceneStrip").querySelectorAll(".scene-chip").forEach((btn) => {
      btn.addEventListener("click", () => toggleScene(btn.dataset.scene || ""));
    });

    $("searchForm").addEventListener("submit", (e) => {
      e.preventDefault();
      if (!hasActiveQueryCondition()) {
        $("qInput").focus();
        return;
      }
      runSearch(1, false);
    });

    $("qInput").addEventListener("input", syncClearBtn);
    $("clearInputBtn").addEventListener("click", () => {
      $("qInput").value = "";
      syncClearBtn();
      $("qInput").focus();
      if (!hasActiveQueryCondition()) showLandingMode();
    });

    $("resetBtn").addEventListener("click", resetAll);
    $("brandHome").addEventListener("click", resetAll);
    const annToggle = $("announcementToggle");
    if (annToggle) {
      annToggle.addEventListener("click", () => {
        state.announcementsExpanded = !state.announcementsExpanded;
        if (state.stats) renderAnnouncements(state.stats.announcements);
      });
    }

    ["dateFrom", "dateTo", "sortFilter", "pageSizeFilter", "fuzzyToggle", "speakerFilterToggle"].forEach((id) => {
      const el = $(id);
      if (el) {
        el.addEventListener("change", () => {
          if (hasActiveQueryCondition()) runSearch(1, false);
        });
      }
    });

    $("bvidFilter").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        runSearch(1, false);
      }
    });

    // 关于 / 作者弹窗
    const openModal = () => $("aboutModal").classList.remove("hidden");
    const closeModal = () => $("aboutModal").classList.add("hidden");
    $("aboutBtn").addEventListener("click", openModal);
    $("footerAboutBtn").addEventListener("click", (e) => { e.preventDefault(); openModal(); });
    $("closeModalBtn").addEventListener("click", closeModal);
    $("aboutModal").addEventListener("click", (e) => {
      if (e.target === $("aboutModal")) closeModal();
    });

    try {
      const [mRes, sRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/members`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/v1/stats?track=1`, { cache: "no-store" })
      ]);
      if (mRes.ok) {
        const mData = await mRes.json();
        state.members = mData.members || [];
        if (mData.site) {
          state.site = { ...state.site, ...mData.site };
        }
      }
      if (sRes.ok) {
        state.stats = await sRes.json();
        lastStatsFetchAt = Date.now();
      }
    } catch (err) {
      // 后端不可达时保留默认成员卡片
    }

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && !state.hasSearched) {
        refreshStats(false);
      }
    });

    renderMemberChips();
    renderLandingStatsAndCards();
    renderAuthorAndGithub();
    updateSearchPlaceholder();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
