"""为 movies 表建立语义向量索引（RAG 建库）。

用法（在 backend 目录下运行）：
    python build_embeddings.py

依赖 .env 中的 EMBEDDING_API_KEY 等配置。会遍历 movies 表，
调用星火 MaaS 嵌入接口，将每部电影的向量写入 movie_embeddings 表。
可重复运行（已存在的 movie_id 覆盖更新，换模型后重跑即可重建索引）。

健壮性：通过 .run/embed.pid 记录自身 PID，并在启动时清理「上一次遗留、
仍在运行的建库进程」，避免重复点击/异常退出后孤儿进程堆积、并发狂刷
星火 MaaS 嵌入接口导致限流/卡死。
"""

import os
import signal

# 项目根（backend 的上一级）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PID_FILE = os.path.join(_ROOT, ".run", "embed.pid")


def _cleanup_prior_build():
    """若 .run/embed.pid 记录了一个仍在运行的旧建库进程，先终止它。"""
    if not os.path.exists(_PID_FILE):
        return
    try:
        with open(_PID_FILE, "r", encoding="utf-8") as f:
            old_pid = int(f.read().strip())
    except Exception:  # noqa: BLE001
        return
    if old_pid and old_pid != os.getpid():
        try:
            os.kill(old_pid, signal.SIGTERM)  # Windows 下映射为 TerminateProcess
            print(f"[清理] 已终止上次遗留的建库进程(PID={old_pid})")
        except ProcessLookupError:
            pass  # 已退出，无需处理
        except Exception as e:  # noqa: BLE001
            print(f"[清理] 终止遗留进程失败（可忽略）：{e}")


def _write_pid():
    os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
    with open(_PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _remove_pid():
    try:
        if os.path.exists(_PID_FILE):
            os.remove(_PID_FILE)
    except Exception:  # noqa: BLE001
        pass


def main():
    _cleanup_prior_build()
    _write_pid()
    try:
        # 延迟导入：先完成 PID 登记与清理，再加载重依赖
        from app.ai.rag import build_movie_embeddings
        import datetime
        print("开始为电影建立语义向量索引……")
        n = build_movie_embeddings()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"完成：已为 {n} 部电影建立/更新语义向量。（重建时间 {ts}）")
    finally:
        _remove_pid()


if __name__ == "__main__":
    main()
