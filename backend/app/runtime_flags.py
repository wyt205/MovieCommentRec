"""运行时开关：控制「自主 Agent 模式」开/关。

设计目的：
- 默认「规则路由模式」（deterministic）：代码抽参 + 代码调工具 + LLM 润色，
  可靠、零幻觉，是日常主路径。
- 打开「自主 Agent 模式」（autonomous）：电影类问题改由 LLM 通过 function calling
  自主选工具、护栏强制先查库，真实展示 function calling + 护栏能力（适合演示/对比/简历）。

用进程内线程安全字典保存，后端单进程即可；启动器只起一个 uvicorn 进程，足够。
改这个开关不需要重启后端——前端/管理端点一下即可实时切换。
"""

import threading

_runtime: dict[str, bool] = {"autonomous": False}
_lock = threading.Lock()


def is_autonomous() -> bool:
    """当前是否处于「自主 Agent 模式」。"""
    with _lock:
        return _runtime["autonomous"]


def set_autonomous(value: bool) -> bool:
    """切换模式，返回设置后的状态。"""
    with _lock:
        _runtime["autonomous"] = bool(value)
        return _runtime["autonomous"]
