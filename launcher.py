# -*- coding: utf-8 -*-
"""
llm-pro 总启动器
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
     「管理端」仍为预留（待实现）。
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

IS_WIN = sys.platform.startswith("windows")
DEVNULL = subprocess.DEVNULL

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
        self.root.title("llm-pro 总启动器")
        self.root.geometry("660x640")
        self.root.resizable(True, True)

        self.procs = {"backend": None, "frontend": None, "admin": None, "tmdb": None, "embed": None}
        self.running = {"backend": False, "frontend": False, "admin": False, "tmdb": False, "embed": False}

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        ttk.Label(
            self.root, text="llm-pro 总启动器", font=("Microsoft YaHei", 16, "bold")
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
        self._service_row(svc, "前端 (Vue :5173)", "frontend",
                          FRONTEND_URL, "进入前端页")

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
        ttk.Button(row, text=enter_text, command=lambda: self._open(enter_url)).pack(side="left", padx=2)

    @staticmethod
    def _draw_dot(canvas, color):
        canvas.delete("all")
        canvas.create_oval(2, 2, 12, 12, fill=color, outline="")

    def _set_status(self, key, running):
        self.running[key] = running
        dot = getattr(self, f"dot_{key}", None)
        if dot:
            self._draw_dot(dot, "green" if running else "red")

    def _open(self, url):
        import webbrowser
        webbrowser.open(url)
        self.log_msg(f"[打开浏览器] {url}")

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
        if self.running.get(key):
            self.log_msg(f"[提示] {key} 已在运行")
            return
        if key == "backend":
            self._spawn_backend()
        elif key == "frontend":
            self._spawn_frontend()

    def _spawn(self, args, cwd, tag, shell=False, env=None):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0
        return subprocess.Popen(
            args, cwd=cwd, shell=shell, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            creationflags=flags,
        )

    @staticmethod
    def _backend_env() -> dict:
        """返回传给后端 / 建库子进程的干净环境。

        剔除可能由启动器进程残留的 LLM_*/EMBEDDING_*/DATABASE_URL/TMDB_API_KEY，
        强制子进程从 backend/.env 重新读取最新配置——这样改完 .env 后，
        只需在启动器里「停止后端 → 启动后端」即可生效，不必重启整个启动器。
        """
        env = dict(os.environ)
        for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                  "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL",
                  "DATABASE_URL", "TMDB_API_KEY"):
            env.pop(k, None)
        return env

    def _spawn_backend(self):
        # 启动前先清掉 8000 端口上可能残留的旧后端，避免多个进程抢同一端口
        killed = self._kill_by_port(BACKEND_PORT)
        if killed:
            self.log_msg(f"[清理] 已释放 {killed} 个占用后端端口(:{BACKEND_PORT})的残留进程")
        env = self._backend_env()
        try:
            proc = self._spawn(
                [BACKEND_PYTHON, "-m", "uvicorn", "app.main:app",
                 "--reload", "--port", str(BACKEND_PORT)],
                BACKEND_DIR, "后端", env=env,
            )
            self.procs["backend"] = proc
            self._set_status("backend", True)
            self.log_msg(f"[后端] 启动中... (python={BACKEND_PYTHON})")
            threading.Thread(target=self._reader, args=(proc, "后端"), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[后端][错误] {e}")

    def _spawn_frontend(self):
        npm = self._find_npm()
        if not npm:
            self.log_msg("[前端][错误] 未找到 npm，请确认 Node.js 已安装并在 PATH 中")
            return
        # 启动前先清掉 5173 端口上可能残留的旧前端（vite 若发现端口被占会自动 +1 到 5174，
        # 导致启动器「进入前端页」指向的 5173 其实是旧进程），避免堆叠
        killed = self._kill_by_port(FRONTEND_PORT)
        if killed:
            self.log_msg(f"[清理] 已释放 {killed} 个占用前端端口(:{FRONTEND_PORT})的残留进程")
        try:
            proc = self._spawn(f"{npm} run dev", FRONTEND_DIR, "前端", shell=True)
            self.procs["frontend"] = proc
            self._set_status("frontend", True)
            self.log_msg("[前端] 启动中... (npm run dev)")
            threading.Thread(target=self._reader, args=(proc, "前端"), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[前端][错误] {e}")

    @staticmethod
    def _find_npm():
        try:
            subprocess.run("npm --version", shell=True,
                           capture_output=True, timeout=5)
            return "npm"
        except Exception:  # noqa: BLE001
            return None

    def start_all(self):
        """一键启动前后端：先清理前端/后端端口上的全部残留进程，再依次启动。"""
        self.log_msg("🚀 一键启动前后端：开始清理残留端口进程...")
        self._cleanup_ports()
        self.start_service("backend")
        self.start_service("frontend")

    def _cleanup_ports(self) -> int:
        """一键启动前清理：杀掉前端(5173)与后端(8000)、管理端(5180)端口上的所有残留进程，
        并在日志展示共释放了多少个进程。返回释放总数。"""
        b = self._kill_by_port(BACKEND_PORT)
        f = self._kill_by_port(FRONTEND_PORT)
        a = self._kill_by_port(ADMIN_PORT)
        total = b + f + a
        if total:
            self.log_msg(
                f"[清理] 共释放 {total} 个残留进程"
                f"（后端 :{BACKEND_PORT} → {b} 个 / 前端 :{FRONTEND_PORT} → {f} 个"
                f" / 管理端 :{ADMIN_PORT} → {a} 个）"
            )
        else:
            self.log_msg("[清理] 端口干净，无残留进程，直接启动 ✅")
        return total

    def _reader(self, proc, tag):
        try:
            for line in proc.stdout:
                self.root.after(0, self.log_msg, f"[{tag}] {line.rstrip()}")
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        rc = proc.wait()
        self.root.after(0, self._on_proc_exit, tag, rc)

    def _on_proc_exit(self, tag, rc):
        if tag == "向量库":
            self.running["embed"] = False
            self.procs["embed"] = None
            ok = rc == 0
            self.embed_status_label.config(
                text="状态：已构建 ✅" if ok else "状态：失败 ❌",
                foreground="#1a7f37" if ok else "#b42318",
            )
            if ok:
                self.log_msg("[向量库] 构建完成！agent 现已支持『按主题/剧情语义找片』。")
            else:
                self.log_msg("[向量库] 构建失败：请检查 backend/.env 是否配置 EMBEDDING_API_KEY，"
                             "以及星火 MaaS 嵌入接口是否可达。")
            return
        key = {"后端": "backend", "前端": "frontend", "管理端": "admin", "TMDb": "tmdb"}.get(tag)
        if key and self.running.get(key):
            self._set_status(key, False)
            self.log_msg(f"[{tag}] 进程已退出 (code={rc})")
        if tag == "TMDb":
            self.tmdb_progress.stop()

    # --------------------------------------------------------------- 停止
    def stop_service(self, key):
        if key == "backend":
            # 按端口清掉 8000 上所有残留后端（可能不止本启动器启动的那一个，
            # 比如之前手动在终端起过、或多次点击启动堆叠的进程）
            self._kill_by_port(BACKEND_PORT)
        proc = self.procs.get(key)
        if proc and self.running.get(key):
            self._kill_tree(proc.pid)
            self.procs[key] = None
            self._set_status(key, False)
            self.log_msg(f"[{key}] 已停止")
        else:
            self.log_msg(f"[{key}] 未在运行")

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
        """返回当前正在 LISTEN 指定 TCP 端口的进程 PID 列表（Windows 用 netstat）。"""
        pids: list[int] = []
        if not IS_WIN:
            return pids
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"],
                stderr=DEVNULL, text=True, timeout=10,
            )
            for line in out.splitlines():
                cols = line.split()
                # 形如: TCP 127.0.0.1:8000 0.0.0.0:0 LISTENING 1234
                if len(cols) >= 5 and cols[3] == "LISTENING":
                    addr = cols[1]
                    port_str = addr.rsplit(":", 1)[-1].rstrip("]")  # 精确取端口，避免 18000 误匹配 8000；兼容 IPv6 [::1]:8000
                    if port_str == str(port):
                        try:
                            pids.append(int(cols[4]))
                        except ValueError:
                            pass
        except Exception:  # noqa: BLE001
            pass
        return pids

    def _kill_by_port(self, port: int) -> int:
        """强制杀掉所有监听指定端口的进程（含其子进程），返回被清理的进程数量。"""
        pids = self._pids_on_port(port)
        for pid in pids:
            try:
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=DEVNULL, stderr=DEVNULL,
                )
            except Exception:  # noqa: BLE001
                pass
        return len(pids)

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
            threading.Thread(target=self._tmdb_reader, args=(proc,), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[TMDb][错误] {e}")
            self.tmdb_progress["value"] = 0

    def _tmdb_reader(self, proc):
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                m = re.search(r"\[STEP\]\s*(\d+)/(\d+)", line)
                if m:
                    self.root.after(0, self._set_tmdb_progress,
                                    int(m.group(1)), int(m.group(2)))
                self.root.after(0, self.log_msg, f"[TMDb] {line}")
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        rc = proc.wait()
        self.root.after(0, self._on_proc_exit, "TMDb", rc)

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
        self.running["embed"] = True
        self.embed_status_label.config(text="状态：构建中…", foreground="#b58100")
        self.log_msg("[向量库] 开始切片 + 嵌入（调用星火 MaaS 嵌入 API）…")
        try:
            proc = self._spawn(
                [BACKEND_PYTHON, "build_embeddings.py"],
                BACKEND_DIR, "向量库", env=env,
            )
            self.procs["embed"] = proc
            threading.Thread(target=self._embed_reader, args=(proc,), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.running["embed"] = False
            self.embed_status_label.config(text="状态：失败 ❌", foreground="#b42318")
            self.log_msg(f"[向量库][错误] {e}")

    def _embed_reader(self, proc):
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self.root.after(0, self.log_msg, f"[向量库] {line}")
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        rc = proc.wait()
        self.root.after(0, self._on_proc_exit, "向量库", rc)

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
            threading.Thread(target=self._admin_reader, args=(proc,), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"[管理端][错误] {e}")

    def _admin_reader(self, proc):
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self.root.after(0, self.log_msg, f"[管理端] {line}")
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        rc = proc.wait()
        self.root.after(0, self._on_proc_exit, "管理端", rc)

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
