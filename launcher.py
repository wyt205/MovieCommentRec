# -*- coding: utf-8 -*-
"""
智影 总启动器
=================================================================
一个桌面 GUI（Tkinter，Python 标准库，零额外依赖），用于统一管理
前后端 / 爬虫 / 预留功能。

功能：
  1. 单独启动「后端 FastAPI」或「前端 Vue」
  2. 一键启动前后端
  3. 状态灯：绿点 = 运行中，红点 = 已停止
  4. 「进入」按钮：前端 → 浏览器打开 localhost:5173；后端 → 打开 /docs
  5. 选择性停止（停前端 / 停后端 / 停全部）
  6. 数据库（MySQL）连通测试：启动前先点「测试连接」排查账号/库/服务问题，
     并提供「打开 .env」一键生成/编辑数据库连接串
  7. 数据爬取（TMDb 公开 API）+ 实时日志 + 进度条：填 Key / 数量 / 关键词即可，自动下载海报
  8. RAG 语义向量库：新增「建立 / 重建语义向量库」按钮，一键跑 backend/build_embeddings.py
     （为 movies 表全部电影切片 + 调用星火 MaaS 嵌入接口生成向量，存入 MySQL 的 movie_embeddings 表）。
     「管理端」已激活（独立 admin/ 静态站，可单独 gitignore）。
  9. 智能诊断：后端启动失败 / 前端 vite 缺失时，日志区自动给出排查建议

运行方式（推荐在 llm-pro 虚拟环境中执行）：
    conda activate llm-pro
    python launcher.py
若直接双击（base 环境），启动器会自动探测 llm-pro 环境的 python 路径来跑后端。
"""
import os
import sys
import signal
import subprocess
import tempfile
import threading
from urllib.parse import urlparse

import re

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# 子进程（如 vite/npm）输出带 ANSI 颜色转义码，Tk 文本框无法渲染，需剥离
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Windows 上 sys.platform 为 "win32"（不是 "windows"），必须用 startswith("win")
# 此前误写成 startswith("windows")，导致 IS_WIN 在 Windows 上恒为 False，
# 整个 Windows 分支（端口查杀 / taskkill 进程树）被全部跳过，
# 于是「停止 / 停止全部」在 Windows 上根本杀不掉任何进程。
IS_WIN = sys.platform.startswith("win")
DEVNULL = subprocess.DEVNULL

# 端口→PID 映射缓存：避免启停时反复起 powershell 进程导致卡顿、进程堆积。
# 整个启停流程（stop_all 对 3 个服务、启动前后端各 1 次）只需真正起 **1 次**
# PowerShell，其余调用全部命中此缓存。
_PORT_CACHE = {}
_PORT_CACHE_T = 0.0
_PORT_CACHE_TTL = 1.5  # 秒


def _all_listening(force: bool = False) -> "dict[int, list[int]]":
    """一次性返回「监听端口 -> 占用进程 PID 列表」的映射，并缓存 _PORT_CACHE_TTL 秒。

    这是「停止/启动卡顿、powershell.exe 进程堆积」的根因修复：此前
    _pids_on_port 每次都 subprocess 起一个 powershell.exe、且 _kill_by_port 还
    多轮扫描，单次启停会 spawn 十几个 powershell 进程、耗时 9~15s。改为一次性
    拿全部监听端口 + 短时缓存后，整个启停流程最多只起 1 次 PowerShell。

    优先 netstat（本机实测 40ms 出结果且能正确返回监听行），PowerShell 仅兜底。
    """
    import time
    global _PORT_CACHE, _PORT_CACHE_T
    now = time.time()
    if not force and _PORT_CACHE and now - _PORT_CACHE_T < _PORT_CACHE_TTL:
        return _PORT_CACHE
    mapping: "dict[int, list[int]]" = {}
    if IS_WIN:
        # 1) netstat 主力（原生、极快——本机实测 40ms 出结果且能正确返回监听行）。
        #    注意：必须用 `netstat -ano`（不过滤协议），不能写 `netstat -ano -p TCP`——
        #    后者只查 IPv4 监听器，会漏掉 TCPv6 监听器。vite 默认绑 `localhost`，
        #    本机 Node 把 localhost 解析成 IPv6 的 ::1，于是 vite 监听在
        #    `::1:5173`（TCPv6）；若只查 TCP，前端端口判定会恒为 False，
        #    导致「检测不到前端已起、一直橙色转红」。用 -ano 同时覆盖 IPv4/IPv6。
        try:
            raw = subprocess.check_output(
                ["netstat", "-ano"],
                stderr=DEVNULL,
            )
            out = raw.decode("mbcs", errors="replace")
            for line in out.splitlines():
                cols = line.split()
                if len(cols) >= 5 and cols[3] == "LISTENING":
                    port_str = cols[1].rsplit(":", 1)[-1].rstrip("]")
                    if port_str.isdigit() and cols[4].isdigit():
                        mapping.setdefault(int(port_str), []).append(int(cols[4]))
        except Exception:  # noqa: BLE001
            pass
    if not mapping:
        # 2) PowerShell 兜底（仅当 netstat 完全无输出，极老系统；慢 ~1.2s/次，平时走不到）
        try:
            ps = ("Get-NetTCPConnection -State Listen "
                  "| Select-Object -Property LocalPort, OwningProcess")
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                stderr=DEVNULL, text=True, timeout=15,
            )
            for line in out.splitlines():
                cols = line.split()
                if len(cols) >= 2 and cols[0].isdigit() and cols[1].isdigit():
                    mapping.setdefault(int(cols[0]), []).append(int(cols[1]))
        except Exception:  # noqa: BLE001
            pass
    _PORT_CACHE.clear()
    _PORT_CACHE.update(mapping)
    _PORT_CACHE_T = time.time()
    return _PORT_CACHE


def _port_in_use(port: int) -> bool:
    """端口是否被占用（基于一次性缓存的 _all_listening 判断，可靠且不堆进程）。

    用于「启动前 / 停止前」短路：仅当端口**真被占用**才调 _kill_by_port 清理；
    正常启停（端口空闲）只触发 1 次 netstat（~40ms，且 1.5s 内命中缓存），
    几乎无感、不堆进程，从而回到「修改前」的流畅体验。
    """
    return int(port) in _all_listening(force=True)

# ---------- 路径配置 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
ADMIN_DIR = os.path.join(ROOT, "admin")

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
ADMIN_URL = "http://localhost:5180"
API_DOCS_URL = "http://localhost:8000/docs"

BACKEND_PORT = 8000
FRONTEND_PORT = 5173
ADMIN_PORT = 5180


def resolve_backend_python() -> str:
    """优先用运行本脚本的环境；否则探测 llm-pro conda 环境。"""
    exe = sys.executable
    if "llm-pro" in exe or os.path.join("envs", "llm-pro") in exe:
        return exe
    candidates = [
        r"D:\Study\Anaconda\envs\llm-pro\python.exe",
        os.path.expanduser(r"~\.conda\envs\llm-pro\python.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return exe  # 退回当前环境，由用户保证依赖已装


BACKEND_PYTHON = resolve_backend_python()


# TMDb 官方类型目录（与 backend/app/core/genres.py 单一数据源一致），
# 用于爬取面板「类型」下拉。导入失败则降级为空列表（类型下拉不可用，其余功能正常）。
try:
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)
    from app.core.genres import TMDB_GENRES as _TMDB_GENRES
except Exception as _e:  # noqa: BLE001
    _TMDB_GENRES = []
    print(f"[launcher] 无法加载 TMDb 类型目录（『类型』下拉将不可用）：{_e}")
_GENRE_BY_NAME = {g["name"]: g["id"] for g in _TMDB_GENRES}
# 爬取模式：展示名 → TMDb 参数
MODE_MAP = {"热门": "popular", "高分": "top_rated", "按类型发现": "discover"}
# discover 排序：展示名 → TMDb sort_by
SORT_MAP = {"热度": "popularity.desc", "评分": "vote_average.desc", "最新上映": "primary_release_date.desc"}


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("智影 总启动器")
        self.root.geometry("660x640")
        self.root.resizable(True, True)

        self.procs = {"backend": None, "frontend": None, "admin": None, "tmdb": None, "embed": None}
        self.running = {"backend": False, "frontend": False, "admin": False, "tmdb": False, "embed": False}
        # 看门狗重启冷却时间戳：避免后端起不来的情况下陷入重启风暴
        self._watchdog_last = {}
        # 各子进程日志落盘路径（.run/<service>.log）；GUI 端尾随文件，不回灌子进程
        self._log_paths = {}
        # 前端实际监听端口（vite 可能因 :5173 被占而自动递增到 5174+），
        # 供「进入前端页」动态打开真实端口，也用于检测时探测范围。
        self.frontend_port = FRONTEND_PORT
        self.frontend_url = FRONTEND_URL

        self._build_ui()
        # 健康看门狗：后端进程「假死」（端口仍被占用但 /health 无响应）时
        # 自动重启，避免 GUI 绿灯亮着、实际已卡死却查不出问题的老毛病。
        threading.Thread(target=self._health_watchdog, daemon=True).start()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        ttk.Label(
            self.root, text="智影 总启动器", font=("Microsoft YaHei", 16, "bold")
        ).pack(pady=(10, 4))
        ttk.Label(
            self.root,
            text=f"后端运行环境：{BACKEND_PYTHON}",
            font=("Consolas", 8),
        ).pack()

        # 服务管理
        svc = ttk.LabelFrame(self.root, text="服务管理")
        svc.pack(fill="x", padx=12, pady=4)
        self._service_row(svc, "后端 (FastAPI :8000)", "backend",
                          API_DOCS_URL, "进入 API 文档")
        # 前端 enter_url 传 None：用 _open_service 动态打开实际探测到的端口
        # （vite 可能自动跳到 5174+，按钮需跟着跳）。
        self._service_row(svc, "前端 (Vue :5173)", "frontend",
                          None, "进入前端页")

        # 一键 / 停止全部
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=12, pady=6)
        ttk.Button(bar, text="🚀 一键启动前后端", command=self.start_all).pack(side="left", padx=4)
        ttk.Button(bar, text="⏹ 停止全部", command=self.stop_all).pack(side="left", padx=4)

        # 数据库连通测试
        dbf = ttk.LabelFrame(self.root, text="数据库（MySQL）")
        dbf.pack(fill="x", padx=12, pady=4)
        self.db_dot = tk.Canvas(dbf, width=14, height=14)
        self.db_dot.pack(side="left", padx=(6, 6))
        self._draw_dot(self.db_dot, "gray")
        ttk.Label(dbf, text="连接状态：").pack(side="left")
        self.db_status_label = ttk.Label(dbf, text="未测试", foreground="#666666")
        self.db_status_label.pack(side="left", padx=(0, 8))
        ttk.Button(dbf, text="测试连接", command=self.test_db).pack(side="left", padx=4)
        ttk.Button(dbf, text="打开 .env", command=self.open_env).pack(side="left", padx=4)

        # 数据爬取（TMDb 公开 API，无反爬，可下载海报）
        cf = ttk.LabelFrame(self.root, text="数据爬取（TMDb 公开 API · 支持中文 · 自动下载海报到本地）")
        cf.pack(fill="x", padx=12, pady=4)
        row0 = ttk.Frame(cf)
        row0.pack(fill="x", padx=4, pady=2)
        ttk.Label(row0, text="API Key 或 Token(二选一):").pack(side="left")
        self.tmdb_key = ttk.Entry(row0, width=42)
        self.tmdb_key.pack(side="left", padx=2, fill="x", expand=True)
        row0b = ttk.Frame(cf)
        row0b.pack(fill="x", padx=4, pady=2)
        ttk.Label(row0b, text="代理(可选):").pack(side="left")
        self.tmdb_proxy = ttk.Entry(row0b, width=42)
        self.tmdb_proxy.pack(side="left", padx=2, fill="x", expand=True)
        row1 = ttk.Frame(cf)
        row1.pack(fill="x", padx=4, pady=2)
        ttk.Label(row1, text="数量:").pack(side="left")
        self.tmdb_count = ttk.Entry(row1, width=6)
        self.tmdb_count.insert(0, "20")
        self.tmdb_count.pack(side="left", padx=2)
        ttk.Label(row1, text="搜索关键词(可选):").pack(side="left", padx=(10, 0))
        self.tmdb_search = ttk.Entry(row1, width=18)
        self.tmdb_search.pack(side="left", padx=2, fill="x", expand=True)
        # —— 爬取条件（多条件组合）——
        row1b = ttk.Frame(cf)
        row1b.pack(fill="x", padx=4, pady=2)
        ttk.Label(row1b, text="模式:").pack(side="left")
        self.tmdb_mode = tk.StringVar(value="热门")
        self.cb_mode = ttk.Combobox(row1b, textvariable=self.tmdb_mode,
                                    values=list(MODE_MAP.keys()), state="readonly", width=12)
        self.cb_mode.pack(side="left", padx=2)
        ttk.Label(row1b, text="类型:").pack(side="left", padx=(8, 0))
        genre_values = ["全部"] + [g["name"] for g in _TMDB_GENRES]
        self.tmdb_genre = tk.StringVar(value="全部")
        self.cb_genre = ttk.Combobox(row1b, textvariable=self.tmdb_genre,
                                     values=genre_values, state="readonly", width=10)
        self.cb_genre.pack(side="left", padx=2)
        ttk.Label(row1b, text="排序:").pack(side="left", padx=(8, 0))
        self.tmdb_sort = tk.StringVar(value="热度")
        self.cb_sort = ttk.Combobox(row1b, textvariable=self.tmdb_sort,
                                    values=list(SORT_MAP.keys()), state="readonly", width=10)
        self.cb_sort.pack(side="left", padx=2)

        row1c = ttk.Frame(cf)
        row1c.pack(fill="x", padx=4, pady=2)
        ttk.Label(row1c, text="年份(可选):").pack(side="left")
        self.tmdb_year = ttk.Entry(row1c, width=8)
        self.tmdb_year.pack(side="left", padx=2)
        ttk.Label(row1c, text="最少评分人数(可选):").pack(side="left", padx=(8, 0))
        self.tmdb_minvotes = ttk.Entry(row1c, width=8)
        self.tmdb_minvotes.pack(side="left", padx=2)
        self.tmdb_onlynew = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1c, text="仅爬取新数据(跳过已存在)", variable=self.tmdb_onlynew).pack(side="left", padx=(8, 0))

        # 模式切换时，仅 discover 才启用 类型/排序/年份/投票 控件
        def _on_mode_change(*_):
            discover = MODE_MAP.get(self.tmdb_mode.get()) == "discover"
            state = "readonly" if discover else "disabled"
            self.cb_genre.configure(state=state)
            self.cb_sort.configure(state=state)
            for ent in (self.tmdb_year, self.tmdb_minvotes):
                ent.configure(state="normal" if discover else "disabled")

        self.cb_mode.bind("<<ComboboxSelected>>", _on_mode_change)
        _on_mode_change()  # 初始化：默认「热门」，禁用类型/排序/年份/投票

        row2 = ttk.Frame(cf)
        row2.pack(fill="x", padx=4, pady=2)
        ttk.Button(row2, text="开始爬取", command=self.start_tmdb).pack(side="left", padx=4)
        ttk.Button(row2, text="停止", command=lambda: self.stop_service("tmdb")).pack(side="left", padx=4)
        self.tmdb_progress = ttk.Progressbar(row2, mode="determinate", length=180)
        self.tmdb_progress.pack(side="left", padx=8)
        ttk.Label(
            cf,
            text="用法：只填『API Key 或 Token』一种即可（自动识别 v3/v4），留空则用 backend/.env 里的 TMDB_API_KEY。"
                 "模式：热门/高分/按类型发现；『按类型发现』可再选 类型+排序+年份+最少评分人数 组合（多条件）。"
                 "留空关键词拉『热门电影』，填了关键词则按名搜索（单一条件，优先级最高）。"
                 "勾选『仅爬取新数据』会跳过库中已有影片、避免重复。国内直连 TMDb 常超时，"
                 "请在『代理』框填地址：支持 http / https / socks5。例：http://127.0.0.1:7890（Clash 本地 HTTP）"
                 "或 socks5://127.0.0.1:7891（Clash 本地 SOCKS5）。本机若已开系统代理会自动识别，不用手填。"
                 "没有代理则直连 TMDb 必超时——可买个『机场』订阅拿到的代理 URL 直接粘这里，本机无需装任何 VPN 软件；"
                 "或把爬虫放到有国际网的云/Colab 跑完再导入本地库。",
            font=("Microsoft YaHei", 8), foreground="#666666",
        ).pack(fill="x", padx=4, pady=(0, 4))

        # AI / RAG 语义检索
        af = ttk.LabelFrame(self.root, text="AI 能力（RAG 语义检索）")
        af.pack(fill="x", padx=12, pady=4)
        arow = ttk.Frame(af)
        arow.pack(fill="x", padx=4, pady=4)
        self.embed_status_label = ttk.Label(arow, text="状态：未构建", foreground="#666666")
        self.embed_status_label.pack(side="left", padx=(0, 8))
        ttk.Button(arow, text="建立 / 重建语义向量库",
                   command=self.start_build_embeddings).pack(side="left", padx=4)
        ttk.Button(arow, text="管理端",
                   command=self.start_admin).pack(side="left", padx=4)
        ttk.Label(
            af,
            text="点击后为 movies 表全部电影切片并调用星火 MaaS 嵌入接口生成向量，"
                 "存入 MySQL 的 movie_embeddings 表（即『切片内嵌 MySQL』）。需先到 backend/.env 配置 "
                 "EMBEDDING_API_KEY（与 GLM key 不同）。构建后 agent 才能『按主题/剧情语义找片』。"
                 "可重复点击重建（换嵌入模型后必点）。",
            font=("Microsoft YaHei", 8), foreground="#666666",
        ).pack(fill="x", padx=4, pady=(0, 4))

        # 日志
        lf = ttk.LabelFrame(self.root, text="运行日志")
        lf.pack(fill="both", expand=True, padx=12, pady=6)
        self.log = scrolledtext.ScrolledText(
            lf, height=12, state="disabled", font=("Consolas", 9)
        )
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

        # 日志窗口已就绪，再预填 Key / 代理（这些函数里会写日志）
        self._prefill_tmdb_key()
        self._prefill_proxy()

    def _service_row(self, parent, label, key, enter_url, enter_text):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=3)
        dot = tk.Canvas(row, width=14, height=14)
        dot.pack(side="left", padx=(0, 6))
        self._draw_dot(dot, "red")
        setattr(self, f"dot_{key}", dot)
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Button(row, text="启动", command=lambda: self.start_service(key)).pack(side="left", padx=2)
        ttk.Button(row, text="停止", command=lambda: self.stop_service(key)).pack(side="left", padx=2)
        ttk.Button(row, text=enter_text, command=lambda: self._open_service(key, enter_url)).pack(side="left", padx=2)

    @staticmethod
    def _draw_dot(canvas, color):
        canvas.delete("all")
        canvas.create_oval(2, 2, 12, 12, fill=color, outline="")

    def _set_status(self, key, running):
        self.running[key] = running
        dot = getattr(self, f"dot_{key}", None)
        if dot:
            self._draw_dot(dot, "green" if running else "red")

    def _set_dot(self, key, color):
        """直接把状态灯设为指定颜色（如橙色 '#f0a000' 表示『启动中』），不改动 running 标志。"""
        dot = getattr(self, f"dot_{key}", None)
        if dot:
            self._draw_dot(dot, color)

    def _open(self, url):
        import webbrowser
        webbrowser.open(url)
        self.log_msg(f"[打开浏览器] {url}")

    def _open_service(self, key, url):
        """打开服务页：url 为 None 时（前端）用实际探测到的端口，
        兼容 vite 因 :5173 被占而自动跳到 5174+ 的情况。"""
        target = url if url else (self.frontend_url if key == "frontend" else None)
        if target:
            self._open(target)

    def log_msg(self, msg):
        self._insert(msg)
        self._maybe_hint(msg)

    def _insert(self, msg):
        msg = _ANSI_RE.sub("", msg)  # 去掉 ANSI 颜色码，避免日志乱码
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _maybe_hint(self, msg):
        """根据已知报错关键字追加诊断建议。"""
        if "Access denied" in msg:
            self._insert("[诊断] 数据库账号/密码错误：请编辑 backend/.env 的 DATABASE_URL，"
                         "把占位密码 'password' 改成你的 MySQL 密码。")
        if "'vite'" in msg:
            self._insert("[诊断] 前端依赖未安装：请先在 frontend 目录运行 'npm install'，再启动前端。")
        if "Application startup failed" in msg:
            self._insert("[诊断] 后端启动失败，通常源于上方数据库连接问题。请先用『测试连接』按钮排查。")

    # --------------------------------------------------------------- 启动
    def start_service(self, key):
        # 若仍标记为运行中（多因上次异常退出未复位，或点过启动又点停止），
        # 先复位状态再启动，避免「点了启动却没反应」的卡死现象。
        if self.running.get(key):
            self.log_msg(f"[提示] {key} 仍标记为运行中，先复位状态再启动")
            self.running[key] = False
        if key == "backend":
            self._spawn_backend()
        elif key == "frontend":
            self._spawn_frontend()

    def _log_path_for(self, tag: str) -> str:
        """各子进程日志落盘路径（.run/<service>.log）。"""
        safe = {"后端": "backend", "前端": "frontend", "TMDb": "tmdb",
                "向量库": "embed", "管理端": "admin"}.get(tag, "service")
        d = os.path.join(ROOT, ".run")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{safe}.log")

    def _spawn(self, args, cwd, tag, shell=False, env=None):
        # 关键修复（根治「后端运行约 1 分钟假死、看门狗误重启」）：
        # 子进程 stdout/stderr 重定向到日志文件，而非 PIPE。
        # 旧写法把后端输出经 PIPE 交给 GUI 读取线程；uvicorn 的访问日志在
        # 『事件循环线程』里写 stdout，一旦 GUI 读取线程因 Tk 主线程繁忙而跟不上、
        # 4KB 管道被写满，事件循环线程就阻塞在 write() 上 → 连异步 /health 都无响应
        # → 看门狗误判假死并重启。改为写文件后，子进程写日志永不被 GUI 卡住，
        # 死锁类假死从根上消失；GUI 端由下方 _log_tailer 异步尾随文件、不回灌子进程。
        log_path = self._log_path_for(tag)
        self._log_paths[tag] = log_path
        logf = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0
        return subprocess.Popen(
            args, cwd=cwd, shell=shell, env=env,
            stdout=logf, stderr=subprocess.STDOUT,
            creationflags=flags,
        )

    @staticmethod
    def _backend_env() -> dict:
        """返回传给后端 / 建库子进程的干净环境。

        直接读取 backend/.env 并把其中的 LLM_*/EMBEDDING_*/DATABASE_URL/TMDB_API_KEY
        注入子进程环境，确保子进程**无论用哪个 Python、cwd 在哪**都能稳定拿到最新配置——
        不再依赖「先剔除环境变量、再靠相对路径 .env 重新加载」这条脆弱链路
        （该链路在启动器所用 Python 与 CLI 不同时，会偶发拿不到依赖/配置而静默失败）。
        改完 .env 后，只需在启动器里「停止 → 启动」即可生效。
        """
        env = dict(os.environ)
        env_path = os.path.join(BACKEND_DIR, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k:
                            env[k] = v
            except Exception:  # noqa: BLE001
                pass
        return env

    def _spawn_backend(self):
        # 一键启动 = 纯启动，不再在此杀进程 / 关接口（职责已移到「停止全部」）。
        # 若 :8000 已被占用，uvicorn 会绑定失败，_wait_backend_ready 超时后转红并提示先「停止全部」。
        env = self._backend_env()
        try:
            proc = self._spawn(
                [BACKEND_PYTHON, "-m", "uvicorn", "app.main:app",
                 "--port", str(BACKEND_PORT)],
                BACKEND_DIR, "后端", env=env,
            )
            self.procs["backend"] = proc
            # 立刻把灯置橙（启动中），给点击即时反馈，避免「点了没反应」的错觉；
            # 真正就绪（/health 返回 200）才转绿，失败转红。
            self.running["backend"] = False
            self._set_dot("backend", "#f0a000")
            self.log_msg(f"[后端] 启动中... 等待 :{BACKEND_PORT} 就绪 (python={BACKEND_PYTHON})")
            threading.Thread(target=self._wait_backend_ready, args=(proc,), daemon=True).start()
            threading.Thread(target=self._log_tailer, args=(proc, "后端"), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[后端][错误] {e}")

    def _wait_backend_ready(self, proc):
        """后端健康检测：轮询 /health，直到 200（端口真在监听）或超时。

        解决「一键启动没反应 / 假绿灯」：uvicorn --reload 的 reloader 父进程
        起来 ≠ 端口已监听；若启动失败（MySQL 未起 / 端口被占 / 依赖缺失）
        状态灯却误显示绿。这里用真实 HTTP 探活，成功才置绿。
        """
        import time
        import urllib.request
        url = f"http://127.0.0.1:{BACKEND_PORT}/health"
        deadline = time.time() + 25
        while time.time() < deadline:
            # 仅当本线程跟踪的 proc 仍是「当前后端」时才判定失败——
            # 否则用户点「停止→启动」后，旧线程盯着已死的旧进程会误报「启动失败」。
            if proc.poll() is not None and self.procs.get("backend") is proc:
                # 进程已退出 = 启动失败
                self.root.after(0, self._set_status, "backend", False)
                self.root.after(
                    0, self.log_msg,
                    "[后端][错误] 进程已退出，启动失败！请查看上方日志"
                    "（常见原因：MySQL 未启动 / 端口被占 / 依赖未装）",
                )
                return
            try:
                with urllib.request.urlopen(url, timeout=2):
                    self.root.after(0, self._set_status, "backend", True)
                    self.root.after(0, self.log_msg, f"[后端] 已就绪 ✅ (端口 :{BACKEND_PORT} 监听中)")
                    return
            except Exception:
                time.sleep(1)
        # 超时：仅当本线程仍对应当前后端时才置红，避免误报掩盖新一次成功启动
        if self.procs.get("backend") is proc:
            self.root.after(0, self._set_status, "backend", False)
            self.root.after(0, self.log_msg, f"[后端][超时] 25s 内未就绪，启动可能失败，请查看上方日志")

    # ----------------------------------------------------- 健康看门狗
    def _health_watchdog(self):
        """后台线程：定期探活后端 /health。

        仅针对『端口仍被占用、但 /health 无响应』的假死场景自动重启——
        这是此前反复出现的「绿灯亮着、实际已卡死、切页像断网」的根因。
        若后端已干净退出（端口空），交由 _log_tailer 正常复位，看门狗不动；
        若 MySQL 没起导致后端起不来，端口也是空的，同样不触发，避免重启风暴。
        """
        import time
        import urllib.request
        url = f"http://127.0.0.1:{BACKEND_PORT}/health"
        while True:
            time.sleep(15)
            try:
                if not self.running.get("backend"):
                    continue
                try:
                    with urllib.request.urlopen(url, timeout=4):
                        continue  # 健康，跳过
                except Exception:
                    pass
                # 探活失败：仅当端口仍被占用（=假死而非干净退出）才重启
                if not _port_in_use(BACKEND_PORT):
                    continue
                now = time.time()
                if now - self._watchdog_last.get("backend", 0) < 30:
                    continue
                self._watchdog_last["backend"] = now
                self.log_msg("[看门狗] 检测到后端进程假死（端口仍占用但无响应），准备自动重启…")
                # 切回主线程执行启停，避免跨线程操作 Tk / 子进程
                self.root.after(0, self._watchdog_restart_backend)
            except Exception:  # noqa: BLE001
                pass

    def _watchdog_restart_backend(self):
        """看门狗触发的后端重启（在主线程执行）。"""
        self.stop_service("backend")
        self._spawn_backend()

    def _spawn_frontend(self):
        npm = self._find_npm()
        if not npm:
            self.log_msg("[前端][错误] 未找到 npm，请确认 Node.js 已安装并在 PATH 中")
            return
        # 纯启动：不再在此杀进程 / 关接口（职责已移到「停止全部」）。
        # 注意：用 shell=True 跟踪的是 cmd.exe/npm 包装进程，它退出 ≠ vite 退出；
        # 因此前端「是否就绪 / 是否还活着」一律以端口 :5173 是否监听为准（见
        # _wait_frontend_ready / _log_tailer / _on_proc_exit），避免误判「进程已退出」。
        try:
            proc = self._spawn(f"{npm} run dev", FRONTEND_DIR, "前端", shell=True)
            self.procs["frontend"] = proc
            # 立刻置橙（启动中），给点击即时反馈；真正就绪（:5173 监听）才转绿。
            self.running["frontend"] = False
            self._set_dot("frontend", "#f0a000")
            self.log_msg("[前端] 启动中... (npm run dev)")
            threading.Thread(target=self._wait_frontend_ready, args=(proc,), daemon=True).start()
            threading.Thread(target=self._log_tailer, args=(proc, "前端"), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[前端][错误] {e}")

    def _wait_frontend_ready(self, proc):
        """前端健康检测：对前端端口发起 HTTP 探活（与后端 /health 同源思路，最可靠）。

        关键修复（此前一直橙色 → 超时转红，但前端其实已起，且『之前能检测、现在不行』）：
          - 旧版用 netstat `-p TCP` 判定端口占用，但 vite 默认绑 `localhost`，本机 Node
            把 localhost 解析成 IPv6 的 ::1，于是 vite 监听在 TCPv6 的 `::1:5173`；
            而 `netstat -p TCP` 只查 IPv4，根本看不到它 → `_port_in_use(5173)` 恒为 False。
          - 另一版改用 HTTP 探活时又写死 `127.0.0.1`（IPv4），同样连不上 ::1。
        两个坑叠加，把原本能用的检测搞坏了。现改为：
          - 用 HTTP GET 探活，依次尝试 `localhost` / `[::1]` / `127.0.0.1`
            （覆盖 IPv6 / IPv4 任一绑定，哪个先应答就判就绪，前端只用 5173，不扫其它端口）；
          - 用 ProxyHandler({}) 强制不走系统代理，避免 127.0.0.1/localhost 被代理拦截；
          - 超时放宽到 90s，容纳 vite 冷启动；检测到即置绿。
        """
        import time
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        # 单端口 5173（用户每次启动前都会先关闭，不会跳端口）；
        # 同时试 IPv6(::1) 与 IPv4(127.0.0.1) 两种地址族，兼容 vite 的 ::1 绑定。
        hosts = ["localhost", "[::1]", "127.0.0.1"]
        deadline = time.time() + 90
        while time.time() < deadline:
            # 仅当本线程跟踪的 proc 已不是「当前前端」时才中止本轮（用户中途重启过）
            if proc.poll() is not None and self.procs.get("frontend") is not proc:
                return
            detected = None
            for h in hosts:
                try:
                    opener.open(f"http://{h}:{FRONTEND_PORT}/", timeout=2)
                    detected = h
                    break
                except Exception:
                    continue
            if detected is not None:
                self.frontend_port = FRONTEND_PORT
                self.frontend_url = f"http://localhost:{FRONTEND_PORT}"
                self.root.after(0, self._set_status, "frontend", True)
                self.root.after(
                    0, self.log_msg,
                    f"[前端] 已就绪 ✅ (端口 :{FRONTEND_PORT} 监听中，探测命中 {detected})")
                return
            time.sleep(1)
        # 超时：仅当本线程仍对应当前前端时才置红，避免误报掩盖新一次成功启动
        if self.procs.get("frontend") is proc:
            self.root.after(0, self._set_status, "frontend", False)
            self.root.after(
                0, self.log_msg,
                f"[前端][超时] 90s 内未就绪，启动可能失败，请查看上方日志"
                f"（若 :{FRONTEND_PORT} 被其它程序占用，请先『停止全部』再启动）")

    @staticmethod
    def _find_npm():
        try:
            subprocess.run("npm --version", shell=True,
                           capture_output=True, timeout=5)
            return "npm"
        except Exception:  # noqa: BLE001
            return None

    def start_all(self):
        """一键启动前后端：纯启动（不杀进程、不关接口）。

        职责边界（按用户要求重构）：
        - 一键启动 = 只负责「启动」后端 + 前端；
        - 杀进程 / 释放端口 = 全部交给「停止全部」按钮。
        若端口已被占用，启动会失败（状态灯转红并有提示），此时应先点「停止全部」。
        """
        self.log_msg("🚀 一键启动前后端：开始启动（不清理端口，若提示端口被占用请先点『停止全部』）…")
        self.start_service("backend")
        self.start_service("frontend")

    def _log_tailer(self, proc, tag):
        """异步尾随子进程日志文件，把新增行转发到 GUI 日志区。

        与旧 _reader（读 proc.stdout PIPE）本质不同：这里读的是『文件』，
        不会反向阻塞子进程——子进程写日志永不被 GUI 读取线程卡住。

        关键修复（根治「rc=None 误报失败 + 子进程被遗弃后台卡死」）：
        1) 读日志转发放进独立 try 块——即便转发途中（如后台线程里
           self.root.after 抛异常）出错，也**绝不影响「等待进程真正结束」**；
        2) 无论读日志成败，最终都显式 proc.wait(timeout) 拿真实退出码，
           再回调 _on_proc_exit；不再依赖被异常吞掉后留下的 proc.poll()==None。
        """
        import time  # 本函数内用到 time.sleep 做日志轮询节流，必须局部导入
        log_path = self._log_paths.get(tag)
        _port = {"后端": BACKEND_PORT, "前端": FRONTEND_PORT,
                 "管理端": ADMIN_PORT}.get(tag)
        # 1) 读日志转发（出错只记录、不中止等待）
        if log_path:
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(0, os.SEEK_END)  # 只看启动后的新日志
                    # 结束条件 = 进程已退出「且」端口不再监听；否则继续尾随。
                    # 前端用 shell=True 跟踪的是 npm 包装进程，它会先于 vite 退出，
                    # 但 vite 仍在 :5173 提供服务——只要端口还活着就继续读日志。
                    # proc.poll() 为 None（进程存活）时 or 短路，不查端口，零额外开销。
                    while proc.poll() is None or (_port and _port_in_use(_port)):
                        line = f.readline()
                        if line:
                            if tag == "TMDb":
                                m = re.search(r"\[STEP\]\s*(\d+)/(\d+)", line)
                                if m:
                                    self.root.after(
                                        0, self._set_tmdb_progress,
                                        int(m.group(1)), int(m.group(2)))
                            self.root.after(0, self.log_msg, f"[{tag}] {line.rstrip()}")
                        else:
                            time.sleep(0.2)
            except Exception as e:  # noqa: BLE001
                try:
                    self.root.after(
                        0, self.log_msg,
                        f"[{tag}][tailer] 读日志异常（不影响等待进程结束）：{e!r}")
                except Exception:  # noqa: BLE001
                    pass
        # 2) 等进程真正结束，拿真实退出码（一次性任务给足超时，避免误判）
        timeout = 1800 if tag == "向量库" else 600
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            rc = None
            try:
                self.root.after(
                    0, self.log_msg,
                    f"[{tag}] 进程 {timeout}s 内未结束，已强制终止（疑似卡死，请检查网络/代理）。")
            except Exception:  # noqa: BLE001
                pass
        try:
            self.root.after(0, self._on_proc_exit, tag, rc)
        except Exception:  # noqa: BLE001
            pass

    def _on_proc_exit(self, tag, rc):
        if tag == "向量库":
            self.running["embed"] = False
            self.procs["embed"] = None
            # 真实判定：优先看子进程退出码 rc；但若日志已打出「完成：已为 N 部」，
            # 说明建库脚本确实跑通（此前出现过退出码误判导致误报失败），以日志为准，
            # 避免「实际已成功建库、GUI 却报失败」让用户误以为启动器坏了。
            lp = self._log_paths.get("向量库")
            log_text = ""
            if lp and os.path.exists(lp):
                try:
                    with open(lp, "r", encoding="utf-8", errors="replace") as f:
                        log_text = f.read()
                except Exception:  # noqa: BLE001
                    pass
            built_ok = ("完成：已为" in log_text) or (
                "开始为电影建立语义向量索引" in log_text and "Traceback" not in log_text)
            ok = (rc == 0) or built_ok
            self.log_msg(f"[向量库] 调试：进程退出码 rc={rc}，日志显示建库"
                         f"{'成功' if built_ok else '未完成'} → 判定 {'成功 ✅' if ok else '失败 ❌'}")
            self.embed_status_label.config(
                text="状态：已构建 ✅" if ok else "状态：失败 ❌",
                foreground="#1a7f37" if ok else "#b42318",
            )
            if ok:
                self.log_msg("[向量库] 构建完成！agent 现已支持『按主题/剧情语义找片』。")
            else:
                self.log_msg("[向量库] 构建失败。上方日志已显示真实报错，常见原因：")
                self.log_msg("[向量库] 1) backend/.env 未配置/失效 EMBEDDING_API_KEY；2) 星火 MaaS 接口限流或不可达；3) 启动器所用 Python 缺依赖。")
                # 把日志里最后的真实报错再贴一次，避免只看通用提示而漏掉根因
                if log_text:
                    tail = [l.rstrip() for l in log_text.splitlines() if l.strip()][-12:]
                    for l in tail:
                        self.log_msg(f"[向量库][log] {l}")
            return
        key = {"后端": "backend", "前端": "frontend", "管理端": "admin", "TMDb": "tmdb"}.get(tag)
        if key and self.running.get(key):
            # 前端特例：被跟踪的 npm 包装进程会先于 vite 退出，但 vite 仍在 :5173
            # 提供服务。此时端口仍监听，说明服务活着，不能判为「已退出」。
            if tag == "前端" and _port_in_use(FRONTEND_PORT):
                return
            self._set_status(key, False)
            self.log_msg(f"[{tag}] 进程已退出 (code={rc})")
        if tag == "TMDb":
            self.tmdb_progress.stop()

    # --------------------------------------------------------------- 停止
    def stop_service(self, key):
        # 按端口清掉对应服务监听的所有残留进程：覆盖「本启动器启动的」+「手动/历史遗留的」
        # （之前只有 backend 走这一步，frontend/admin 只杀记录的进程树，会漏掉遗留进程）。
        port = {"backend": BACKEND_PORT, "frontend": FRONTEND_PORT,
                "admin": ADMIN_PORT}.get(key)
        if port and _port_in_use(port):
            killed = self._kill_by_port(port)
            if killed:
                self.log_msg(f"[停止] 已释放 {killed} 个占用端口(:{port})的进程")
        # 再补刀：杀掉本启动器记录跟踪的那个进程树（含其子进程，如 vite/esbuild）
        proc = self.procs.get(key)
        if proc:
            self._kill_tree(proc.pid)
        # 关键修复：无论之前是否在跑、proc 是否为 None，停止后一律复位状态标志
        # 并置红灯，避免「running 卡在 True、灯却不动」导致下次点启动没反应。
        self.procs[key] = None
        self.running[key] = False
        self._set_status(key, False)
        self.log_msg(f"[{key}] 已停止")

    def stop_all(self):
        for k in ("backend", "frontend", "admin", "tmdb", "embed"):
            self.stop_service(k)

    def _kill_tree(self, pid):
        if IS_WIN:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=DEVNULL, stderr=DEVNULL,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _pids_on_port(port: int) -> list[int]:
        """返回当前正在 LISTEN 指定 TCP 端口的进程 PID 列表（基于一次性缓存）。

        具体查询交给模块级 _all_listening()：一次 PowerShell 拿全部监听端口，
        1.5s 内复用缓存，避免启停时反复起 powershell 进程导致卡顿/堆积。
        """
        return list(_all_listening().get(int(port), []))

    def _kill_by_port(self, port: int) -> int:
        """强制杀掉所有监听指定端口的进程（含其子进程），返回被清理的进程数量。

        短路：先用 socket 毫秒级探测 _port_in_use——端口本就空闲时直接返回 0，
        不查端口、不起任何进程，恢复「修改前」0 卡顿、不堆进程的体验；
        仅当端口确被占用才走 netstat 查杀。每轮开头清空端口缓存，
        确保拿到的是实时监听状态，避免「刚杀掉的旧 PID 残留在缓存」造成假阳性计数。
        """
        if not _port_in_use(port):
            return 0
        total = 0
        import time
        for _ in range(2):
            # 每轮开头清空缓存，保证本轮查到的是实时状态（刚杀掉的不会残留）
            global _PORT_CACHE, _PORT_CACHE_T
            _PORT_CACHE.clear()
            _PORT_CACHE_T = 0.0
            pids = self._pids_on_port(port)
            if not pids:
                break
            for pid in pids:
                try:
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=DEVNULL, stderr=DEVNULL,
                    )
                except Exception:  # noqa: BLE001
                    pass
            total += len(pids)
            time.sleep(0.3)
        return total

    # --------------------------------------------------------------- 数据爬取 (TMDb)
    def start_tmdb(self):
        if self.running.get("tmdb"):
            self.log_msg("[TMDb] 已在运行")
            return
        key = self.tmdb_key.get().strip()
        if not key:
            key = self._read_tmdb_key_from_env()
        if not key:
            self.log_msg("[TMDb][错误] 请先填写 TMDb API Key / Token（或写入 backend/.env 的 TMDB_API_KEY）")
            return
        try:
            count = int(self.tmdb_count.get().strip() or "20")
        except ValueError:
            count = 20
        search = self.tmdb_search.get().strip() or None
        proxy = self.tmdb_proxy.get().strip()
        # 爬取条件
        mode = MODE_MAP.get(self.tmdb_mode.get(), "popular")
        genre_id = None
        sort_by = None
        year = None
        min_votes = None
        if mode == "discover":
            gname = self.tmdb_genre.get()
            genre_id = _GENRE_BY_NAME.get(gname) if gname and gname != "全部" else None
            sort_by = SORT_MAP.get(self.tmdb_sort.get(), "popularity.desc")
            ys = self.tmdb_year.get().strip()
            year = int(ys) if ys.isdigit() else None
            ms = self.tmdb_minvotes.get().strip()
            min_votes = int(ms) if ms.isdigit() else None
        only_new = self.tmdb_onlynew.get()
        env = dict(os.environ)
        env["TMDB_API_KEY"] = key
        env["TMDB_COUNT"] = str(count)
        env["TMDB_MODE"] = mode
        if search:
            env["TMDB_SEARCH"] = search
        if mode == "discover":
            if genre_id is not None:
                env["TMDB_GENRE_ID"] = str(genre_id)
            if sort_by:
                env["TMDB_SORT_BY"] = sort_by
            if year:
                env["TMDB_YEAR"] = str(year)
            if min_votes:
                env["TMDB_MIN_VOTES"] = str(min_votes)
        if only_new:
            env["TMDB_ONLY_NEW"] = "1"
        if proxy:
            env["HTTP_PROXY"] = proxy
            env["HTTPS_PROXY"] = proxy
            env["ALL_PROXY"] = proxy
        try:
            proc = self._spawn(
                [BACKEND_PYTHON, "-m", "app.crawler.tmdb"],
                BACKEND_DIR, "TMDb", env=env,
            )
            self.procs["tmdb"] = proc
            self.running["tmdb"] = True
            self.tmdb_progress["maximum"] = count
            self.tmdb_progress["value"] = 0
            if search:
                desc = f"关键词「{search}」"
            elif mode == "discover":
                desc = f"按类型发现(类型={self.tmdb_genre.get()}, 排序={self.tmdb_sort.get()}" \
                       f"{', 年份=' + str(year) if year else ''}" \
                       f"{', 最少投票=' + str(min_votes) if min_votes else ''})"
            elif mode == "top_rated":
                desc = "高分电影"
            else:
                desc = "热门电影"
            if only_new:
                desc += " · 仅爬新数据"
            self.log_msg(f"[TMDb] 启动拉取（{desc}，count={count}）...")
            threading.Thread(target=self._log_tailer, args=(proc, "TMDb"), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[TMDb][错误] {e}")
            self.tmdb_progress["value"] = 0

    def _set_tmdb_progress(self, value, total):
        self.tmdb_progress["maximum"] = total
        self.tmdb_progress["value"] = value

    # ----------------------------------------------- 语义向量库 (RAG 建库)
    def start_build_embeddings(self):
        """一键为 movies 表全部电影切片 + 调用星火 MaaS 嵌入接口生成向量，存入 MySQL。"""
        if self.running.get("embed"):
            self.log_msg("[向量库] 正在构建中，请勿重复点击")
            return
        if not os.path.exists(BACKEND_PYTHON):
            self.log_msg("[向量库][错误] 找不到后端 python 路径，无法构建")
            return
        env = self._backend_env()
        # 清理上次「被遗弃仍运行」的建库进程，避免与本次并发狂刷星火 MaaS
        # 嵌入接口（会触发限流/卡死，导致本次也建库失败）。孤儿常源于早前
        # 旧版启动器在子进程未结束时误报失败、又把引用置空，使进程脱离管控。
        _pid_file = os.path.join(ROOT, ".run", "embed.pid")
        if os.path.exists(_pid_file):
            try:
                with open(_pid_file, "r", encoding="utf-8") as f:
                    old_pid = int(f.read().strip())
                if old_pid and old_pid != os.getpid():
                    try:
                        os.kill(old_pid, signal.SIGTERM)  # Windows 下映射为 TerminateProcess
                        self.log_msg(f"[向量库] 已清理上次遗留的建库进程(PID={old_pid})")
                    except ProcessLookupError:
                        pass  # 已退出
                    except Exception as e:  # noqa: BLE001
                        self.log_msg(f"[向量库][提示] 清理遗留进程失败（可忽略）：{e}")
            except Exception:  # noqa: BLE001
                pass
        self.running["embed"] = True
        self.embed_status_label.config(text="状态：构建中…", foreground="#b58100")
        # 诊断：直接显示本次建库用的是哪个 Python、key 有没有注入，便于失败自查
        self.log_msg(f"[向量库] 调试：python={BACKEND_PYTHON} 存在={os.path.exists(BACKEND_PYTHON)} "
                     f"EMBEDDING_API_KEY 已注入={'EMBEDDING_API_KEY' in env and bool(env.get('EMBEDDING_API_KEY'))}")
        self.log_msg("[向量库] 开始切片 + 嵌入（调用星火 MaaS 嵌入 API）…")
        try:
            proc = self._spawn(
                [BACKEND_PYTHON, "build_embeddings.py"],
                BACKEND_DIR, "向量库", env=env,
            )
            self.procs["embed"] = proc
            threading.Thread(target=self._log_tailer, args=(proc, "向量库"), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.running["embed"] = False
            self.embed_status_label.config(text="状态：失败 ❌", foreground="#b42318")
            self.log_msg(f"[向量库][错误] {e}")

    # ----------------------------------------------------- 管理端 (独立 admin/ 文件夹)
    def start_admin(self):
        """启动管理端静态站点（admin/ 目录，python -m http.server 托管），
        并自动打开浏览器。管理端与 backend/frontend 完全解耦，可单独 gitignore。"""
        if self.running.get("admin"):
            self._open(ADMIN_URL)
            self.log_msg("[管理端] 已在运行，直接打开浏览器")
            return
        if not os.path.isdir(ADMIN_DIR):
            self.log_msg(f"[管理端][错误] 找不到 admin 目录：{ADMIN_DIR}")
            return
        if not os.path.exists(BACKEND_PYTHON):
            self.log_msg("[管理端][错误] 找不到后端 python 路径，无法启动静态服务")
            return
        # 启动前先清掉 5180 端口上的残留进程（避免旧 admin 占端口导致打不开）
        killed = self._kill_by_port(ADMIN_PORT)
        if killed:
            self.log_msg(f"[清理] 已释放 {killed} 个占用管理端口(:{ADMIN_PORT})的残留进程")
        try:
            proc = self._spawn(
                [BACKEND_PYTHON, "-m", "http.server", str(ADMIN_PORT),
                 "--directory", ADMIN_DIR],
                ADMIN_DIR, "管理端",
            )
            self.procs["admin"] = proc
            self.running["admin"] = True
            self.log_msg(f"[管理端] 已启动静态服务：{ADMIN_URL}（python -m http.server）")
            self._open(ADMIN_URL)
            threading.Thread(target=self._log_tailer, args=(proc, "管理端"), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[管理端][错误] {e}")

    # ------------------------------------------------------------- 预留
    @staticmethod
    def not_implemented(name):
        messagebox.showinfo(
            "待实现",
            f"「{name}」功能尚未接入。\n\n后续只需在 launcher.py 的对应方法里补充启动逻辑即可。",
        )

    def _read_tmdb_key_from_env(self):
        """从 backend/.env 读 TMDB_API_KEY（输入留空时回退用）。"""
        env_path = os.path.join(BACKEND_DIR, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("TMDB_API_KEY"):
                            _, _, v = s.partition("=")
                            v = v.strip().strip('"').strip("'")
                            if v:
                                return v
            except Exception:  # noqa: BLE001
                pass
        return None

    def _prefill_tmdb_key(self):
        """若 backend/.env 已配置 TMDB_API_KEY，预填到输入框。"""
        v = self._read_tmdb_key_from_env()
        if v:
            self.tmdb_key.insert(0, v)

    @staticmethod
    def _detect_windows_proxy() -> str | None:
        """读取 Windows 系统代理（设置 → 网络 → 代理）地址；没有则返回 None。"""
        if not sys.platform.startswith("win"):
            return None
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ) as k:
                enabled, _ = winreg.QueryValueEx(k, "ProxyEnable")
                if not enabled:
                    return None
                server, _ = winreg.QueryValueEx(k, "ProxyServer")
                if not server:
                    return None
                # 取 https=... 或 http=... 或裸地址
                if "=" in server:
                    for part in server.split(";"):
                        if part.startswith("https="):
                            return part.split("=", 1)[1].strip()
                        if part.startswith("http="):
                            return part.split("=", 1)[1].strip()
                    return server.split(";")[0].split("=", 1)[-1].strip()
                return server.strip()
        except Exception:  # noqa: BLE001
            return None

    def _prefill_proxy(self):
        """若用户未手动填代理，且系统已开启代理，则自动填入。"""
        if self.tmdb_proxy.get().strip():
            return
        p = self._detect_windows_proxy()
        if p:
            self.tmdb_proxy.insert(0, p)
            self.log_msg(f"[代理] 已自动识别系统代理：{p}")

    # ------------------------------------------------------- 数据库测试
    def open_env(self):
        """打开 backend/.env（不存在则复制 .env.example 并提示）。"""
        env_path = os.path.join(BACKEND_DIR, ".env")
        if not os.path.exists(env_path):
            example = os.path.join(BACKEND_DIR, ".env.example")
            if os.path.exists(example):
                try:
                    import shutil
                    shutil.copyfile(example, env_path)
                    self.log_msg("[.env] 已从 .env.example 复制生成 .env，请编辑其中的数据库密码。")
                except Exception as e:  # noqa: BLE001
                    self.log_msg(f"[.env] 复制失败：{e}")
        try:
            if IS_WIN:
                os.startfile(env_path)
            else:
                os.system(f"open {env_path}")
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[.env] 无法打开文件：{e}（路径：{env_path}）")

    def _read_db_url(self) -> str:
        default = "mysql+pymysql://root:password@localhost:3306/llm_pro?charset=utf8mb4"
        env_path = os.path.join(BACKEND_DIR, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s and not s.startswith("#") and s.startswith("DATABASE_URL"):
                            _, _, val = s.partition("=")
                            val = val.strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:  # noqa: BLE001
                pass
        return default

    def _set_db_status(self, ok: bool, detail: str):
        self._draw_dot(self.db_dot, "green" if ok else "red")
        self.db_status_label.config(
            text=detail, foreground=("#1a7f37" if ok else "#b42318")
        )

    def test_db(self):
        """读取 backend/.env 的 DATABASE_URL，尝试连 MySQL 并报告。"""
        if not os.path.exists(BACKEND_PYTHON):
            self._set_db_status(False, "后端 python 路径无效")
            self.log_msg("[数据库测试] 找不到后端 python 路径，无法测试")
            return
        url = self._read_db_url()
        parsed = urlparse(url)
        host, port = parsed.hostname or "localhost", parsed.port or 3306
        dbname = parsed.path.lstrip("/").split("?")[0] or "(默认库)"
        self.log_msg(f"[数据库测试] 正在连接 {host}:{port} / {dbname} ...")

        script = (
            "import sys, pymysql\n"
            "from urllib.parse import urlparse\n"
            "u = sys.argv[1]\n"
            "p = urlparse(u)\n"
            "host, port, user = p.hostname, p.port or 3306, p.username\n"
            "pw = p.password\n"
            "db = p.path.lstrip('/').split('?')[0]\n"
            "try:\n"
            "    c = pymysql.connect(host=host, port=port, user=user, password=pw, database=db, connect_timeout=5)\n"
            "    with c.cursor() as cur:\n"
            "        cur.execute('SELECT VERSION()')\n"
            "        ver = cur.fetchone()[0]\n"
            "    c.close()\n"
            "    print('OK:' + str(ver))\n"
            "except Exception as e:\n"
            "    print('FAIL:' + repr(e)[:300])\n"
        )
        tmp = os.path.join(tempfile.gettempdir(), "_dbtest_tmp.py")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(script)
            proc = subprocess.run(
                [BACKEND_PYTHON, tmp, url],
                capture_output=True, text=True, timeout=20,
            )
            out = (proc.stdout or proc.stderr or "").strip()
            if out.startswith("OK:"):
                self._set_db_status(True, f"✅ 连接成功 MySQL {out[3:]}")
                self.log_msg(f"[数据库测试] 连接成功！版本：{out[3:]}")
            elif out.startswith("FAIL:"):
                err = out[5:]
                self._set_db_status(False, f"❌ {err}")
                self.log_msg(f"[数据库测试] 连接失败：{err}")
                self._db_hint(err)
            else:
                self._set_db_status(False, f"输出异常：{out[:80]}")
                self.log_msg(f"[数据库测试] 未知输出：{out}")
        except subprocess.TimeoutExpired:
            self._set_db_status(False, "❌ 连接超时（>20s）")
            self.log_msg("[数据库测试] 连接超时，请确认 MySQL 服务已启动")
        except FileNotFoundError:
            self._set_db_status(False, "❌ 无法运行测试脚本")
            self.log_msg(f"[数据库测试] 找不到 python：{BACKEND_PYTHON}")
        except Exception as e:  # noqa: BLE001
            self._set_db_status(False, f"❌ {e}")
            self.log_msg(f"[数据库测试] 出错：{e}")
        finally:
            try:
                os.remove(tmp)
            except Exception:  # noqa: BLE001
                pass

    def _db_hint(self, err: str):
        if "Access denied" in err:
            self._insert("[诊断] 账号或密码错误：编辑 backend/.env 的 DATABASE_URL，"
                         "把占位密码 'password' 改成你的 MySQL root 密码。")
        elif "Unknown database" in err:
            self._insert("[诊断] 数据库不存在：先执行 'mysql -u root -p < sql/init.sql' 创建 llm_pro 库与表。")
        elif "Can't connect" in err or "Connection refused" in err:
            self._insert("[诊断] 无法连接 MySQL：请确认 MySQL 服务已启动，且 host/port 正确（默认 localhost:3306）。")

    def on_close(self):
        self.stop_all()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LauncherApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
