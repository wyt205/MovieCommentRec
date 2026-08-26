"""Agent 装配：用 LangChain 把「大模型 + 工具 + 会话记忆」串成一个【有护栏的自主智能体（guardrailed agent）】。

核心思路（hybrid 架构）：
- 闲聊 / 身份 / 暂缓(defer)：用【确定性代码】直接处理，不依赖弱模型的"自觉"，可靠且零幻觉。
- 电影相关：默认走【代码驱动确定性路由】——规则抽参(_extract_params_rule) + 代码调工具(_route_tool)
  + 模板复述(_respond_result)，100% 可靠、零幻觉、且只多 1 次 LLM 做自然语言润色。
  · 工具仍是 LangChain @tool 定义的 function-calling 工具（智谱 OpenAI 兼容接口调用）；
    区别是由【代码】决定调哪个、传什么参数，而非依赖弱模型 glm-4-flash 的自主决策
    （实测弱模型 function calling 极不稳定：要么不调工具凭记忆答、要么带错参数，导致幻觉/慢）。
  · 「分别推评分最高+热度最高的电影」→ sort="rating,popularity" 自动拆成两次 find_movies。
  · 自主 function-calling agent(_agent_run_with_guardrail*) 仅作为【确定性路径空手而归时的最后兜底】。

这样既保留真实 agent / function-calling 工具链（简历含金量），又把可靠性交给确定性代码——
绝不会因为弱模型偷懒/抽错参而幻觉，且延迟从最坏 5 次 LLM 调用(~50s)降到 1 次(~10s)。

未配置 llm_api_key 时，_make_llm() 会抛 RuntimeError，由路由层转成 503 友好提示，
因此「模型待定」阶段后端也能照常启动、其它功能不受影响。
"""

import json
import re
import time
import threading
from contextvars import ContextVar
from sqlalchemy.sql import func

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.ai.tools import TOOLS, _USER_QUERY_CTX, _detect_genres_from_text
from app.core.config import settings
from app import crud
from app.db.database import SessionLocal
from app.models import AgentTrace, ChatSession, UserPreference
from app.runtime_flags import is_autonomous

SYSTEM_PROMPT = """你是智影影评网站的智能助手。
- 使用提供的工具查询真实数据库，来回答用户关于电影、影评、推荐的问题。
- 只依据工具返回的数据作答，不要编造电影、评分或影评。

【绝对铁律 · 关于电影的事实只能来自工具】
- 你没有任何电影领域的内部知识或记忆。任何具体电影信息（片名、导演、演员、评分、年份、类型、地区、简介、热度）都必须且只能来自工具返回的数据。
- **只要用户的话里出现了某部电影的名字（例如「我想找一部电影叫超能陆战队」「你知道XXX吗」），无论怎么措辞，都必须先调用 get_movie_info_by_name(片名) 去资料库核实，拿到结果后再开口；绝不要用你的训练记忆去介绍、讨论或假装知道任何电影。**
  · 反面教材：用户说「我想找一部电影叫超能陆战队」，你绝不能先回「你想了解它的哪些信息？」——那是凭记忆假装它存在。正确做法是立刻调 get_movie_info_by_name 核实。
  · **即使用问句、也绝不可凭记忆预设电影属性**：例如用户问「你知道超能陆战队吗」，你绝不能先回「超能陆战队是一部动画电影，对吗？」这类带记忆断言的话——这等于先入为主地"编"了它的类型/年份。必须先调工具核实，工具说没找到就直接说"资料库暂无"，绝不用你训练记忆里的"它是动画/某年上映"去猜、去反问、去补充。
  · 只有用户的话「完全不涉及任何具体电影、影评或推荐」时（例如单纯打招呼「你好」「在吗」），才算闲聊，可不调工具、自然回应即可。
- 顺序铁律：先调用工具，再回答；未调用工具就不得给出任何电影片名或事实。
- get_movie_info_by_name 已一次性返回完整资料（导演/演员/简介等），拿到后直接组织答案，**不要为同一部电影再重复调用 get_movie_detail**。

【资料库没有这部电影时 · 要直接、别绕圈子】
- 当 get_movie_info_by_name / search_movies 返回「未找到」时，说明资料库里暂时没有这部电影（可能还没收录）。请**直接、明确**地告诉用户，例如：「资料库里暂时还没有《XXX》这部电影（可能还没收录）。」然后可以友好地问一句「你想看什么类型的？我帮你推荐类似的」。
- **绝不要反过来追问这部电影本身的年份 / 导演 / 主演 / 类型等细节**——你本来也不知道，追问只会绕圈子，让用户觉得答非所问。
- 资料库返回未找到后，**不要**再去调用 semantic_search_movies 试图「确认」它是否存在（语义检索是按主题/感受找电影，不是用来查某部具体片是否在库的）。直接说未收录即可。

【找电影 / 推荐电影】
- 用户提到具体片名想看详情 → 用 get_movie_info_by_name(片名)。
- 用户按条件组合筛选（类型 genre、年份 year、地区 country、排序 sort、条数 limit）→ 统一用 find_movies，把用户提到的维度作为参数组合传入即可，不要为每种问法单独找工具。
  · 例如「评分最高的动作片」→ find_movies(genre="动作", sort="rating")；「2021年上映的中国电影」→ find_movies(year=2021, country="中国")；「按热度排的科幻片」→ find_movies(genre="科幻", sort="popularity")；「评分最高的电影」（没说类型）→ find_movies(sort="rating")。
  · **【多类型交集 · 必遵守】当用户用「A的B」「A+B」「既A又B」这类说法同时提到多个类型时（例如「爱情的动画」「科幻喜剧」「日本的动画」），必须把『每一个』提到的类型都拼进 genre 参数、用逗号隔开传入（取交集，即同时命中），绝不能只取其中一个而把其它的丢掉。** 错误示范：用户说「爱情的动画」，你只传 genre="爱情" → 会返回一堆非动画的爱情片。正确做法：genre="动画,爱情"。同样，「科幻喜剧」→ genre="科幻,喜剧"；「动画 爱情 奇幻」→ genre="动画,爱情,奇幻"。
  · **绝不要自行脑补用户没说的条件**（尤其不要给 genre 乱填类型）。但反过来，用户『说出口的』每个类型都必须保留，一个都不能少。用户没提年份/地区就留空，只用用户真正说出的筛选/排序维度。
  · country 参数接受中文（美国/日本/中国…），工具内部会自动映射，直接传中文即可。
- 若用户用「剧情/主题/情感/氛围」等含义描述想看的电影（如"讲时间循环""结局治愈""深夜一个人看"），优先用 semantic_search_movies 做语义检索。
- 若用户已从其它工具的返回里拿到了确切的数字 id，才用 get_movie_detail(id)；**绝不要自己编造 movie_id**，没有确切 id 就改用片名工具。

【数量不足时的诚实原则 · 绝不凑数、绝不强编】
- 工具结果里若出现了系统生成的「（注：资料库里符合条件的电影目前共 M 部…未能凑满你想要的 N 部）」这类**明确短缺提示**，你必须**如实**告诉用户：「资料库里这类电影目前只有 M 部，都列给你了」，然后停下，**绝不可**为了凑够 N 部去凭训练记忆编造电影、片名、评分或年份——多编哪怕一部也是严重幻觉，是绝对错误。**注意：工具结果开头写的「找到 X 部」只是本次返回条数（受 limit 影响），并不代表库里只有这么多；没有上述系统短缺提示时，你【绝不要】自行声明「库里只有 X 部」。**
- 优先顺序铁律：**永远先检索资料库，把资料库里符合条件的全部列完**；只有资料库不足、且你明确告知用户「以下是资料库【之外】的电影、仅供参考、并非本站数据」时，才可以补充推荐库外的电影，并且不得把它们伪装成资料库里的电影。
- 反面教材：用户要「5 部冒险动画」，资料库只找到 4 部 → 正确做法是列出这 4 部并说明「资料库里只有 4 部」；错误做法是自己编出第 5 部《XXX》来凑数。

【礼貌回应 / 结束语 · 不续推、不调工具】
- 当用户只是表达感谢、肯定或结束对话（例如「谢谢」「感谢」「好的，感谢」「不错」「赞」「辛苦了」「拜拜」），**只需简短友好地回应**（如「不客气～有需要随时找我🙂」），**绝对不要**再去调用任何工具、也**不要**继续推荐电影或生成新的片单。这类消息不是新的观影需求，不要把它误当成「再推荐一部」。
- 一句话原则：用户没提出新的电影/推荐/影评需求时，就闲聊式收尾，别自作主张地续上推荐。

【多轮对话重要规则】
- 用户在后续对话中「更正或补充」了片名/描述（例如上一轮说「奇幻大冒险」，这一轮说「奇幻变身大冒险呢」），**必须把用户最新给出的词作为工具参数，绝不可沿用上一轮的旧词**。
- 用户可能只记得片名片段或记错字（如「奇幻大冒险」实际是《奇幻变身大冒险》）。get_movie_info_by_name / search_movies 已内置模糊匹配，直接把你理解的片名传入即可；若返回为空，可换一个关键词重试，但不要硬不承认「未找到」。"""

# ---------------------------------------------------------------------------
# 意图识别（代码级强制工具调用，不依赖弱模型"自觉"）
# 弱模型（glm-4-flash 免费档）在「先聊天、不调工具」上有强惯性，纯 prompt 护栏压不住；
# 这里用确定性代码把"电影相关问题"强制绑到对应工具，杜绝凭记忆瞎答 / 多轮被带歪。
# ---------------------------------------------------------------------------
_THANKS_KW = ["谢谢", "感谢", "多谢", "好的", "好嘞", "赞", "辛苦", "拜拜", "不错", "ok", "OK"]
_GREET_KW = ["你好", "您好", "在吗", "hi", "hello", "嗨", "哈喽", "在不在"]
_NAME_KW = ["知道", "了解", "认识", "叫", "这部", "那部", "《", "》", "关于", "讲的是",
            "导演", "主演", "剧情", "简介", "上映", "哪年", "谁演", "看的"]
_RECOMMEND_KW = ["推荐", "找一些", "找几部", "有哪些", "有没有", "想看", "类似",
                 "什么类型", "题材", "排行", "评分最高", "热度最高", "帮我找",
                 "给我找", "来一部", "来几部", "按", "有啥", "有什么"]
# 身份类问题：弱模型（glm-4-flash 免费档）对「你是谁/请问你是」时而答对时而把问题当问候忽略，
# 故用确定性代码直接返回固定自我介绍，不交给模型自由发挥（与 defer 同理，绕开弱模型不稳定性）。
# 注意：不能放裸「你的名字」——它是著名电影《你的名字》的片名，会撞车；问助手名字用「你的名字是什么」。
_IDENTITY_KW = ["你是谁", "你是？", "你是?", "你是什么", "请问你是", "你叫什么", "你叫啥",
                "你的名字是什么", "你的身份", "你是干嘛的", "你是做什么的", "你是什么人",
                "你谁啊", "介绍一下你自己", "介绍下你自己", "你是什么助手", "你是什么机器人"]
# 电影信号词：消息里出现这些，说明在聊电影，绝不应判定为「身份闲聊」。
# 专门治「你知不知你的名字」这类情况——「你的名字」既是身份短语又是电影名，
# 但只要带《》/知道/了解/看过/导演等电影信号，就优先当电影处理。
_MOVIE_SIGNAL_KW = ["《", "》", "电影", "这部", "那部", "影片", "片子", "知道", "了解", "认识",
                    "知不知", "知不知道", "看过", "看没", "听过", "搜", "找", "推荐",
                    "导演", "主演", "剧情", "简介", "上映", "评分", "年份", "类型", "影评"]

# 暂缓信号：用户明确表示「稍后再给要求 / 先别急着推荐 / 等我说」时，不该调工具去查电影，
# 而应自然回应、等他开口。这是确定性规则（不放回模型自由发挥），避免模型"自作主张"先喷电影。
# 注意用词要精准，避免误伤「再给我推荐」这类真正的推荐请求（它不含下面的任何词）。
_DEFER_KW = ["一会儿", "稍后", "先别急", "先不急", "等我说", "等下", "等一下", "待会", "待会儿",
             "还没想好", "还没说", "不急着", "别急着", "你先别", "我还没", "你别急"]


def _is_defer(message: str) -> bool:
    m = message.lower()
    return any(k in m for k in _DEFER_KW)


def _classify_intent(message: str) -> str:
    """返回 'chat' / 'identity' / 'movie_name' / 'recommend' / 'movie_related'。
    顺序：chat → identity(带电影信号抑制) → name → recommend → movie_related。
    - identity 放在 name 之前：避免「你叫什么名字」里的"叫"被 _NAME_KW 误判成查片名；
      同时用 _MOVIE_SIGNAL_KW 抑制，保证「你知道《你是谁》吗」仍走查库而非身份闲聊。
    """
    m = message.lower()
    if any(k in m for k in _THANKS_KW) or any(k in m for k in _GREET_KW):
        return "chat"
    # 身份问题：命中身份词、且不含任何电影信号词时，才判为「问助手自己」
    if any(k in m for k in _IDENTITY_KW) and not any(k in m for k in _MOVIE_SIGNAL_KW):
        return "identity"
    if any(k in m for k in _NAME_KW):
        return "movie_name"
    if any(k in m for k in _RECOMMEND_KW):
        return "recommend"
    return "movie_related"


# —— 会话历史（DB-backed）——
# 每次对话都落库到 chat_sessions 表，因此切换页面 / 重启前后端都不会丢失，
# 也支撑前端「对话记录列表」（可继续聊 / 删除）。
# 兼容 LangChain 的 BaseChatMessageHistory 接口，_run_chat 内 store.add_* 调用无需改动。
def _message_to_dict(m) -> dict:
    role = "user" if m.type == "human" else "assistant"
    return {"role": role, "text": m.content or "", "ts": time.time()}


def _dict_to_message(d: dict):
    return HumanMessage(content=d.get("text", "")) if d.get("role") == "user" else AIMessage(content=d.get("text", ""))


class DBChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    def _row(self, db):
        return db.query(ChatSession).filter_by(session_id=self.session_id).first()

    @property
    def messages(self):
        db = SessionLocal()
        try:
            row = self._row(db)
            if not row or not row.messages:
                return []
            return [_dict_to_message(m) for m in row.messages]
        finally:
            db.close()

    def add_message(self, message) -> None:
        db = SessionLocal()
        try:
            row = self._row(db)
            if row is None:
                row = ChatSession(session_id=self.session_id, messages=[], title="新对话")
                db.add(row)
                db.flush()
            msgs = list(row.messages or [])
            msgs.append(_message_to_dict(message))
            row.messages = msgs
            row.updated_at = func.now()
            # 首次出现用户消息时，自动用其前 24 字作为会话标题
            if (not row.title or row.title == "新对话") and message.type == "human":
                row.title = (message.content or "新对话")[:24]
            db.commit()
        finally:
            db.close()

    def add_user_message(self, text: str) -> None:
        self.add_message(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        self.add_message(AIMessage(content=text))

    def clear(self) -> None:
        db = SessionLocal()
        try:
            row = self._row(db)
            if row:
                row.messages = []
                db.commit()
        finally:
            db.close()

# —— 埋点层（管理端可观测性的地基）——
# 每次对话一个 trace 对象，挂在 contextvar 上，供 _safe_invoke_tool / 护栏 写入工具调用链，
# 对话结束统一落库到 agent_traces 表。管理端的「日志 / 缓存命中 / 护栏使用率」都读这张表。
_TRACE_CTX: ContextVar = ContextVar("agent_trace", default=None)


def _safe_reset(token):
    """安全重置 ContextVar；吞掉跨上下文（如测试客户端 worker 线程、某些 ASGI 服务器
    把生成器挪到别的上下文执行）可能抛出的 ValueError，避免流式响应尾部误报 error 事件。"""
    if token is None:
        return
    try:
        token.var.reset(token)
    except (ValueError, RuntimeError):
        pass


class _Trace:
    """单次对话的可观测数据收集器。"""

    def __init__(self):
        self.tool_calls: list[dict] = []
        self.used_guardrail: bool = False
        self.cache_hit: bool = False
        self.mode: str = "deterministic"  # deterministic=规则路由主路径；autonomous=LLM 自主 function calling 主路径


# 进程内响应缓存：同一问题短期复问直接复用答案（命中即记 cache_hit）。
# 这是管理端「缓存命中率」指标的真实来源；用锁保证多线程（uvicorn 线程池）安全。
_RESP_CACHE: dict[str, str] = {}
_RESP_CACHE_LOCK = threading.Lock()


def _record_trace(session_id: str, query: str, intent: str, tr: "_Trace", answer: str, latency_ms: int):
    """尽力把本次对话的可观测数据落库；任何异常都吞掉，绝不影响正常对话。"""
    try:
        db = SessionLocal()
        try:
            db.add(AgentTrace(
                session_id=session_id,
                query=query,
                intent=intent,
                tool_calls=tr.tool_calls,
                answer=answer,
                latency_ms=latency_ms,
                used_guardrail=tr.used_guardrail,
                cache_hit=tr.cache_hit,
            ))
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass


def _get_history(session_id: str) -> BaseChatMessageHistory:
    """返回 DB-backed 会话历史（每次 add_* 都落库到 chat_sessions 表）。"""
    return DBChatMessageHistory(session_id)


# —— 长期记忆（用户偏好）——
# 轻量规则抽取（不额外调 LLM，省 token 且快）：识别用户消息里的类型喜好，
# 持久化到 user_preferences 表；system_prompt() 会把偏好注入，让 agent 推荐时主动参考。
def _load_user_prefs_text() -> str:
    try:
        db = SessionLocal()
        try:
            rows = db.query(UserPreference).all()
            if not rows:
                return ""
            parts = []
            for r in rows:
                vals = r.value or []
                if not vals:
                    continue
                if r.key == "fav_genres":
                    parts.append("喜欢类型：" + "、".join(vals))
                elif r.key == "disliked_genres":
                    parts.append("不喜欢类型：" + "、".join(vals))
            return "【用户偏好·主动参考】" + "；".join(parts) if parts else ""
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        return ""


def _extract_prefs(message: str) -> None:
    """规则抽取用户类型喜好，高置信才写入；任何异常都吞掉，绝不影响正常对话。"""
    pos = any(w in message for w in ["喜欢", "爱看", "最爱", "钟爱", "偏爱", "迷", "爱"])
    neg = any(w in message for w in ["不喜欢", "讨厌", "反感", "厌恶", "别推", "不要", "不爱"])
    if not (pos or neg):
        return
    genres_found = _detect_genres_from_text(message)
    if not genres_found:
        return
    try:
        db = SessionLocal()
        try:
            for g in set(genres_found):
                key = "fav_genres" if pos else "disliked_genres"
                row = db.query(UserPreference).filter_by(key=key).first()
                cur = set(row.value or []) if row else set()
                cur.add(g)
                if row:
                    row.value = list(cur)
                else:
                    db.add(UserPreference(key=key, value=list(cur)))
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass


def system_prompt() -> str:
    """基础 SYSTEM_PROMPT + 用户长期偏好（若有）。每次对话读一次 DB，开销极小。"""
    pref = _load_user_prefs_text()
    return SYSTEM_PROMPT + ("\n\n" + pref if pref else "")


def _make_llm():
    """构造大脑 LLM（参数提取 / 闲聊共用）。未配置 key 时抛 RuntimeError（由路由层兜底）。"""
    if not settings.llm_api_key:
        raise RuntimeError("未配置 LLM_API_KEY，Agent 未启用。请在 .env 中设置 llm_api_key。")
    return ChatOpenAI(
        model=settings.llm_model or "glm-4-flash",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        temperature=0.3,
        # 上限 + 重复惩罚：弱模型（glm-4-flash 免费档）偶发“重复列举”死循环，
        # 导致答案又长又臭、还拖慢首字。硬性限制输出长度，并降低重复概率。
        # 注：repetition_penalty 是智谱在 OpenAI 兼容接口上的专有字段，openai SDK
        # 会对其做强类型校验、拒绝陌生顶层参数，故必须用 extra_body 透传（而非
        # 直接写成顶层参数）。extra_body 是 openai SDK 官方提供的“透传非标准字段”通道。
        max_tokens=700,
        extra_body={"repetition_penalty": 1.2},
        # 超时与重试：防止智谱侧慢响应/挂起把后端线程池线程也卡死
        # （与 DB 死连接同理，一旦 LLM 调用无限挂起，会拖垮整个后端）。
        # 60s 对 glm-4-flash 这类小模型足够；超过则说明异常，宁可快速失败。
        timeout=60,
        max_retries=1,
    )


def _tool_by_name(name: str):
    for t in TOOLS:
        if t.name == name:
            return t
    return None


def _safe_int(v, default: int = 0) -> int:
    """把模型抽出来的 year/limit 安全地转成 int（模型偶尔会抽成字符串或 0/None）。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# 确定性片名识别：从用户【原话】里认出已知电影片名。
# 弱模型（glm-4-flash）对「你的名字」这类歧义短语有强偏见——总把它理解成「问助手你叫什么名字」，
# 导致抽参抽飞（把整句当片名）、判成闲聊（不查库）、或响应层无视工具结果自答身份。
# 故用代码做 DB 子串匹配（与类型词自动抽取同一思路），只要原话里出现某部已知片名就强制按该电影查库，
# 不依赖弱模型去「理解」片名。
_TITLE_CACHE: list[str] | None = None  # 缓存 DB 全部片名（首次查询后复用）


def _load_titles() -> list[str]:
    global _TITLE_CACHE
    if _TITLE_CACHE is not None:
        return _TITLE_CACHE
    db = SessionLocal()
    try:
        rows = crud.get_movies(db, limit=1000)
        _TITLE_CACHE = [m.title for m in rows if m.title]
    finally:
        db.close()
    return _TITLE_CACHE


def _find_title_in_text(text: str) -> str | None:
    """若用户原话（去标点后）包含某部已知电影的片名，返回 DB 里的准确片名；否则 None。"""
    cleaned = text.replace("《", "").replace("》", "").replace("。", "").replace(" ", "")
    for orig in _load_titles():
        t = orig.replace("。", "").replace(" ", "")
        if t and t in cleaned:
            return orig  # 返回 DB 准确片名（含原标点），交给 get_movie_info_by_name 做模糊匹配
    return None


# 参数提取提示词：让模型把用户问题转成结构化参数（模型只做"抽取"，不做"是否查库"的决策）
_EXTRACT_PROMPT = """你是一个参数提取器。根据用户的问题，提取查询电影数据库所需的参数，只返回一段 JSON（不要任何额外文字、不要 markdown 代码块）。

判定与字段：
- mode="movie_info"：用户明显在问「某部具体电影」本身（如「你知道X吗」「X讲什么」「X的导演/演员/剧情」「X是动画吗」）。填 name（尽量还原准确片名，如「超能陆战队」「指环王」）。
- mode="find"：用户按条件找/推荐电影（如「推荐动画」「评分最高的动作片」「2021日本电影」「冒险题材的动画」）。填：
    genre：类型，多个用逗号隔开（如 动画,冒险），没有则空字符串
    year：年份数字，没有则 0
    country：国家/地区中文（如 美国/日本），没有则空字符串
    sort：排序 rating(评分)/popularity(热度)/year(年份)/release(上映日期)。**若用户要「分别按不同维度各推几部」（如「分别推一个评分最高和一个热度最高的电影」），把多个维度用英文逗号写在 sort 里（如 sort="rating,popularity"），并把 limit 设为每个维度想要的条数（「一个」→limit=1）。** 没有则空字符串
    limit：想要的条数，没有则 0
- mode="semantic"：用户按「剧情/主题/情感/氛围」描述想看的电影（如「讲时间循环」「结局治愈」「深夜一个人看」），没有具体片名也没有明确类型筛选。填 query（用户原意描述）。
- mode="chat"：纯闲聊/问候/感谢（如「你好」「谢谢」「好的」），无需查库。
- defer：布尔值。仅当用户明确表示「稍后再给具体要求 / 还没想好 / 先别急着推荐 / 等我说 / 我一会儿会给你要求 / 你先别急」时为 true；只要用户已经给出了可查询的具体条件（类型/年份/地区/片名/主题），就为 false。注意：用户说「推荐几个电影」但紧接着说「我一会儿会给你要求」，说明此刻还不该查，defer=true。

示例：
用户：「帮我找一些冒险题材的动画」→ {{"mode":"find","defer":false,"name":"","genre":"动画,冒险","year":0,"country":"","sort":"","limit":0}}
用户：「你知道超能陆战队吗」→ {{"mode":"movie_info","defer":false,"name":"超能陆战队","genre":"","year":0,"country":"","sort":"","limit":0}}
用户：「讲时间循环的电影」→ {{"mode":"semantic","defer":false,"name":"","genre":"","year":0,"country":"","sort":"","limit":0,"query":"讲时间循环"}}
用户：「谢谢」→ {{"mode":"chat","defer":false,"name":"","genre":"","year":0,"country":"","sort":"","limit":0}}
用户：「你能给我推荐几个电影吗，我一会儿会给你要求」→ {{"mode":"find","defer":true,"name":"","genre":"","year":0,"country":"","sort":"","limit":0}}
用户：「你先别急，我还没说找什么样的」→ {{"mode":"find","defer":true,"name":"","genre":"","year":0,"country":"","sort":"","limit":0}}
用户：「分别推给我一个评分最高和一个热度最高的电影」→ {{"mode":"find","defer":false,"name":"","genre":"","year":0,"country":"","sort":"rating,popularity","limit":1}}

用户问题：{msg}
"""


def _extract_params(message: str) -> dict:
    """用模型把用户问题抽取成结构化参数（确定性路由的依据）。失败时回退到保守默认值。"""
    default = {"mode": "find", "name": "", "genre": "", "year": 0,
               "country": "", "sort": "", "limit": 0, "query": "", "defer": False}
    try:
        llm = _make_llm()
        resp = llm.invoke(_EXTRACT_PROMPT.format(msg=message))
        text = resp.content or ""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            merged = dict(default)
            merged.update({k: v for k, v in data.items() if k in default})
            return merged
    except Exception:  # noqa: BLE001
        pass
    return default


def _extract_params_rule(message: str) -> dict:
    """确定性参数抽取（不依赖 LLM）：覆盖绝大多数「找电影/推荐」句式，速度快、零幻觉、
    不依赖弱模型是否会抽 JSON。作为【电影意图主路径】的参数来源，替代不稳定的 LLM 抽取。

    返回字段与 _extract_params 完全一致，便于 _route_tool 直接消费。
    """
    m = message
    default = {"mode": "find", "name": "", "genre": "", "year": 0,
               "country": "", "sort": "", "limit": 0, "query": "", "defer": _is_defer(m)}
    # 1) 片名优先：原话里出现已知电影 → 强制按该电影查库
    title = _find_title_in_text(m)
    if title:
        default["mode"] = "movie_info"
        default["name"] = title
        return default
    # 1.5) 显式书名号《》内的片名：即便库内暂无收录，也按「具体电影」走
    #      get_movie_info_by_name（而非退化成 find_movies 泛化筛选），
    #      这样「你知道《X》吗」会得到精准的「未找到与「X」相关的电影」，
    #      工具链也更一致（所有「问某部具体电影」都走 get_movie_info_by_name）。
    bm = re.search(r"《([^》]+)》", m)
    if bm:
        default["mode"] = "movie_info"
        default["name"] = bm.group(1).strip()
        return default
    # 2) 类型（确定性识别，不看历史）
    genres = _detect_genres_from_text(m)
    default["genre"] = ",".join(genres)
    # 3) 年份
    ym = re.search(r"(\d{4})\s*年", m)
    if ym:
        default["year"] = int(ym.group(1))
    # 4) 地区
    for c in ["中国", "美国", "日本", "韩国", "英国", "法国", "德国", "印度",
              "泰国", "中国香港", "中国台湾", "俄罗斯"]:
        if c in m:
            default["country"] = c
            break
    # 5) 排序维度（支持「分别按评分+热度各推」→ 多维度）
    sorts = []
    if any(k in m for k in ["评分最高", "评分", "按评分", "最高分", "打分高", "分高"]):
        sorts.append("rating")
    if any(k in m for k in ["热度最高", "热度", "最火", "按热度", "受欢迎", "看的人多"]):
        sorts.append("popularity")
    if any(k in m for k in ["最新", "年份最高", "按年份", "近年", "近期", "新上映"]):
        sorts.append("year")
    if any(k in m for k in ["上映日期", "上映"]):
        sorts.append("release")
    # 去重保序
    default["sort"] = ",".join(dict.fromkeys(sorts))
    # 6) 条数
    lm = re.search(r"(\d+)\s*[部个]", m)
    if lm:
        default["limit"] = int(lm.group(1))
    elif any(k in m for k in ["几部", "哪些", "哪几部", "有什么", "有哪些", "来几部", "推荐几", "推荐一些", "有哪些电影"]):
        default["limit"] = 5
    elif any(k in m for k in ["全部", "所有", "都给我", "都列", "都列出来"]):
        default["limit"] = 20
    # 7) 语义检索：无具体类型/排序、且是「主题/情感/氛围」描述
    if not genres and not default["sort"] and any(
        k in m for k in ["治愈", "时间循环", "深夜", "一个人", "孤独", "感人", "温暖",
                        "讲一个", "关于", "主题", "氛围", "情感", "适合"]
    ):
        default["mode"] = "semantic"
        default["query"] = m
    return default


# 排序维度 → 中文标签（用于向响应层说明每个分组是按什么排的）
_SORT_LABEL = {
    "rating": "评分最高",
    "popularity": "热度最高",
    "year": "年份",
    "release": "上映日期",
}


def _route_tool(params: dict, message: str, tr=None) -> str:
    """根据结构化参数，在代码里直接调用对应工具，返回工具的真实文本结果。
    message 用于注入 contextvar（供 find_movies 做类型词自动抽取兜底）。
    tr 显式传入用于埋点（记录工具调用链，修复流式异步下记丢的问题）。

    支持「多排序维度」：sort 以逗号分隔多个维度时，分别调用 find_movies 并分块返回，
    从而能正确满足「分别推一个评分最高和一个热度最高」这类诉求（两者往往是不同电影，
    此前只查一次会把同一部电影包装成两个，造成「评分最高=热度最高」的离谱结果）。
    """
    token = _USER_QUERY_CTX.set(message)
    try:
        mode = params.get("mode", "find")
        if mode == "movie_info" and params.get("name"):
            tool = _tool_by_name("get_movie_info_by_name")
            if tr is not None:
                tr.tool_calls.append({"name": "get_movie_info_by_name", "args": {"name": params["name"]}})
            return str(tool.invoke({"name": params["name"]}))
        if mode == "semantic":
            tool = _tool_by_name("semantic_search_movies")
            if tr is not None:
                tr.tool_calls.append({"name": "semantic_search_movies", "args": {"query": params.get("query") or message, "top_k": 5}})
            return str(tool.invoke({"query": params.get("query") or message, "top_k": 5}))

        # 默认 / find：按条件筛选
        # 注意：必须传原始默认值（0 / ""），不要传 None——
        # find_movies 的 Pydantic schema 要求 year 为 int、genre/country/sort 为 str，
        # 传 None 会直接 ValidationError；工具内部已自行做 year/limit or 默认 的处理。
        tool = _tool_by_name("find_movies")
        genre = params.get("genre") or ""
        year = _safe_int(params.get("year"))
        country = params.get("country") or ""
        requested_limit = _safe_int(params.get("limit"))
        per_limit = requested_limit if requested_limit > 0 else 5

        # 多排序维度：拆成多条查询，每条按各自维度排序
        sorts = [s.strip() for s in (params.get("sort") or "").split(",") if s.strip()]
        if not sorts:
            sorts = [""]  # 默认按评分降序

        blocks = []
        for s in sorts:
            if tr is not None:
                tr.tool_calls.append({"name": "find_movies", "args": {"genre": genre, "year": year, "country": country, "sort": s, "limit": per_limit}})
            raw = str(tool.invoke({
                "genre": genre,
                "year": year,
                "country": country,
                "sort": s,
                "limit": per_limit,
            }))
            label = _SORT_LABEL.get(s, "默认") if s else "评分最高（默认）"
            blocks.append(f"【排序：{label}】\n{raw}")
        # 诚实告知：仅当用户「明确要 N 部」、且资料库「真实不足 N 部」时才补一句，
        # 且数字用工具结果里的「真实总条数（共 X 部符合条件）」——不受 limit 截断影响，
        # 避免报出错误数量。用户要的数量已满足（如要 2 部已列出 2 部）时，绝不画蛇添足补数量声明。
        if requested_limit > 0 and blocks:
            m_tot = re.search(r"共 (\d+) 部符合条件", blocks[-1])
            true_total = int(m_tot.group(1)) if m_tot else 0
            if 0 < true_total < requested_limit:
                blocks[-1] += (
                    f"\n（注：资料库里符合条件的电影目前共 {true_total} 部，已全部列出，"
                    f"未能凑满你想要的 {requested_limit} 部。）"
                )
        return "\n\n".join(blocks)
    finally:
        _USER_QUERY_CTX.reset(token)


# 把工具的「真实结果」用自然、会"听人话"的中文讲给用户（事实仍 100% 来自工具，零幻觉）
_RESPOND_TEMPLATE = """你是智影影评网站的智能助手，正在和用户多轮聊电影。

【铁律 · 只能基于工具结果说话】
- 你只能依据下面【工具结果】里给出的信息来谈论电影。禁止编造任何电影名、评分、年份、类型、地区、简介。
- 若【工具结果】是「未找到 / 没有符合条件的电影」，就如实、友好地告诉用户资料库里暂时没有这部电影（可能还没收录），并可以友好地问一句「你想看什么类型的？我帮你推荐类似的」。
- 若【工具结果】是一组电影，就用自然、亲切的口吻介绍这批电影（可加一句引导语，例如「给你挑了几部符合要求的电影：」），不要生硬复述原始格式；保留片名、评分、类型等关键信息即可。
- **禁止自行编造数量声明**：你**绝不可以**自己冒出「资料库里这类电影目前只有 X 部，都列给你了」之类的句子。只有当【工具结果】里出现了系统生成的「（注：资料库里符合条件的电影目前共 M 部…未能凑满你想要的 N 部）」这类**明确短缺提示**时，你才需要把「资料库里这类电影目前只有 M 部，都列给你了」的意思**如实转达**给用户（M 以提示里的数字为准）。用户要的数量已经满足时（例如用户要 2 部、你也列出了 2 部），直接自然介绍电影即可，**不要画蛇添足补一句数量声明**。绝不要为了凑数去编造资料库里没有的片名/评分/年份；若想补充资料库【之外】的电影，必须先明确标注「（资料库外 · 仅供参考）」，且不得伪装成资料库数据。优先做法：先如实交差资料库里的电影。
- 若【工具结果】是一段「单部电影资料」（以《片名》开头、含导演/演员/评分/简介等字段），你必须介绍【那部电影】本身。特别注意：即使用户的话里出现了「你的名字」这类措辞，只要工具结果是一份电影资料，就说明用户在问那部【电影】（例如《你的名字。》），你只能介绍该电影；**绝不要**把它理解成在问「助手你叫什么名字」，也**绝不要**输出任何关于「助手身份 / 助手名字 / 我是谁」的内容。
- 若【工具结果】包含多个以「【排序：xxx】」开头的分组，说明是按不同维度（如评分最高 / 热度最高）分别查到的电影。请【分别、逐一】介绍每个分组，并明确点出该组是按什么维度排的（如「这是评分最高的一部」「这是热度最高的一部」）；不同分组往往是不同的电影，绝对不要把它们混为一谈、也不要只介绍其中一个而漏掉其它分组。
- 【工具结果】里的「【排序：xxx】」只是内部分组标记，组织答案时不要原样照抄这些标记，用自然说法带出（如「按评分排，最高的是…」「按热度排，最高的是…」）。若某组末尾出现了系统生成的「（注：资料库里…共 N 部…未能凑满你想要的 M 部）」这类**明确短缺提示**，才自然地转达给用户（如「不过资料库里这类电影目前只有 N 部，都列给你了」，数字以提示为准）；【工具结果】里没有此类提示时，不要自己添加任何数量声明。
- 绝对不要输出任何工具调用语法（如 function_name(...)）。只输出一段自然中文回复。

【对话上文】
{history}

【用户刚才的问题】
{question}

【工具结果】
{result}
"""

# 用户说"稍后再给要求 / 先别急 / 还没想好"时，不查库，自然回应等他开口
_DEFER_TEMPLATE = """你是智影影评网站的智能助手。用户刚刚表示「稍后再给具体要求 / 还没想好 / 先别急着推荐 / 等我说」。

请只用自然、友好的中文简短回应：表示你明白了，会等他给要求；并可以友好地问一句他想看什么类型、题材、地区或年份，你再帮他挑。
绝对不要现在去查电影、也不要推荐任何具体电影。不要输出工具调用语法。

【对话上文】
{history}

【用户刚才的话】
{question}
"""


def _fmt_history(history) -> str:
    if not history:
        return "(无上文)"
    out = []
    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = getattr(msg, "content", "") or ""
        out.append(f"{role}：{content}")
    return "\n".join(out)


def _llm_reply(system_text: str, history, question: str) -> str:
    llm = _make_llm()
    msgs = [SystemMessage(content=system_text)] + list(history) + [HumanMessage(content=question)]
    resp = llm.invoke(msgs)
    return resp.content or ""


def _llm_reply_stream(system_text: str, history, question: str):
    """与 _llm_reply 同入参，但用 stream=True 逐 token 产出（yield 文本片段）。"""
    llm = _make_llm()
    msgs = [SystemMessage(content=system_text)] + list(history) + [HumanMessage(content=question)]
    for chunk in llm.stream(msgs):
        c = getattr(chunk, "content", "")
        if c:
            yield c


def _respond_chat(message: str, history) -> str:
    return _llm_reply(system_prompt(), history, message)


# 身份类问题：固定自介，确定性返回（不交给弱模型自由发挥，避免「请问你是」被当问候忽略）
_IDENTITY_REPLY = (
    "我是智影影评网站的智能助手🙂 我可以帮你查电影资料、按类型 / 年份 / 地区"
    "推荐电影，或者聊聊影评。你想看什么类型的电影，尽管告诉我～"
)


def _respond_identity(message: str, history) -> str:
    return _IDENTITY_REPLY


# 纯问候固定寒暄（确定性返回，不调 LLM，首问即 ~0ms，无需依赖响应缓存）
_GREET_REPLY = (
    "你好呀～我是智影影评网站的智能助手🙂 "
    "想看什么电影、或者想了解哪部片子，尽管告诉我～"
)

# 纯问候词后的常见语气词/标点，用于判断「是否仅为问候」（去除后应几乎为空）
_GREET_PARTICLES = ["啊", "呀", "呢", "哦", "哈", "嘿", "哟", "喂", "嘛", "啦",
                    "呗", "嗯", "诶", "在", "吗", "？", "?", "！", "!", "。"]

def _is_pure_greeting(message: str) -> bool:
    """判断消息是否为「纯问候」（如「你好」「你好啊」「在吗」「hi」）。
    纯问候不需要调 LLM，直接返回固定寒暄即可（省一次 LLM 往返，~0ms）。
    含实质内容的闲聊（如「讲个笑话」「今天天气怎么样」）返回 False，照常走 LLM。"""
    m = message.lower()
    if not any(k in m for k in _GREET_KW):
        return False
    rest = m
    for kw in _GREET_KW:
        rest = rest.replace(kw, "")
    for p in _GREET_PARTICLES:
        rest = rest.replace(p, "")
    # 去掉空白与标点后，若仍剩实义字符（中文/英文/数字），说明不止是问候
    rest = re.sub(r"[\s\W_]+", "", rest)
    return rest == ""


def _respond_defer(message: str, history) -> str:
    prompt = (_DEFER_TEMPLATE.replace("{history}", _fmt_history(history))
                              .replace("{question}", message))
    return _llm_reply(prompt, [], message)


def _respond_result(message: str, raw_result: str, history) -> str:
    prompt = (_RESPOND_TEMPLATE.replace("{history}", _fmt_history(history))
                                .replace("{question}", message)
                                .replace("{result}", raw_result))
    return _llm_reply(prompt, [], message)


# ---------------------------------------------------------------------------
# 有护栏的自主工具调用循环（guardrailed agent）
# 让模型通过原生 function calling 自主决定调哪个工具、并可多步串联；
# 代码护栏保证「涉及电影就必须先调工具」，弱模型偷懒凭记忆答时强制重试；
# 重试耗尽仍未调工具 → 退回确定性路由兜底（仍零幻觉）。
# ---------------------------------------------------------------------------

def _build_hint_messages(message: str) -> list:
    """把「类型词自动抽取 / 已知片名识别 / 多维度分别推荐」三个确定性安全网，
    组装成 SystemMessage 提示列表（空列表表示无需注入）。供自主工具循环与流式版本共用。"""
    genre_hint = _detect_genres_from_text(message)
    title_hint = _find_title_in_text(message)
    hints = []
    if title_hint:
        hints.append(
            f"用户原话里出现了已知电影片名《{title_hint}》。若用户是在问这部电影，"
            f"必须用 get_movie_info_by_name(name='{title_hint}') 核实，不要用 find_movies 去搜。"
        )
    if genre_hint:
        hints.append(
            f"用户原话里识别到的电影类型词（必须【全部】作为 genre 传入、用逗号隔开取交集，"
            f"例如 genre='动画,冒险'）：{', '.join(genre_hint)}。绝不可只取其中一个而丢掉其它。"
        )
    if ("分别" in message or "各" in message) and any(
        k in message for k in ["评分最高", "热度最高", "评分", "热度", "排行"]
    ):
        hints.append(
            "若用户要「分别按不同维度（如评分最高、热度最高）各推几部」，你必须【分别调用多次 find_movies】——"
            "每次只传一个 sort 维度（例如一次 sort='rating'、一次 sort='popularity'），不要只调一次。"
            "不同维度往往对应不同电影，最后要逐一介绍每个维度查到的电影。"
        )
    if not hints:
        return []
    hint_text = "【自动识别提示 · 必须遵循】" + " ".join(hints)
    return [SystemMessage(content=hint_text)]


_MAX_AGENT_STEPS = 3  # 允许模型多步串联工具；不宜过大，否则弱模型在自主循环里空转、首字延迟飙升
_GUARDRAIL_MSG = (
    "⚠️ 你还没有调用任何工具。根据系统铁律，只要用户的话涉及具体电影、推荐或影评，"
    "你必须先调用数据库工具核实，绝不能用你的训练记忆回答。请立刻调用合适的工具"
    "（get_movie_info_by_name / find_movies / semantic_search_movies）。"
)


def _safe_invoke_tool(tc: dict, tr=None) -> str:
    """执行模型选定的工具，返回工具结果的字符串；任何异常都被兜住，避免循环崩掉。

    tr 显式传入（不再依赖 _TRACE_CTX 上下文变量），避免流式异步消费时上下文丢失、
    导致工具调用链 / 护栏标记记丢（埋点表显示「工具链=— 护栏=否」的元凶）。
    """
    name = tc.get("name", "")
    args = tc.get("args", {}) or {}
    # 埋点：把本次工具调用记进当前 trace（供管理端「日志 / 工具调用链」可视化）
    if tr is not None:
        tr.tool_calls.append({"name": name, "args": args})
    tool = _tool_by_name(name)
    if not tool:
        return f"错误：未知工具 {name}"
    try:
        return str(tool.invoke(args))
    except Exception as e:  # noqa: BLE001
        return f"工具 {name} 调用出错：{e}"


def _sanitize_tool_call(message: str, tc: dict) -> dict:
    """修正弱模型在自主循环里抽错的工具参数。

    根因：弱模型（glm-4-flash）会把【上一轮对话】的类型/过滤条件带进当前请求
    （如上轮问了「冒险题材的动画」，这轮问「最高评分电影」却仍带 genre=动画）。
    这里用【只看当前这句话】的确定性逻辑强制纠正：
      - genre：只用当前句里识别到的类型词（_detect_genres_from_text，不看历史）；
               若当前句无类型词且不含「不再局限/所有电影/不限」等解除词 → genre 留空（不继承历史）。
      - 若当前句含解除词（不再局限 / 所有电影 / 不限 / 去掉…限制）→ 强制 genre=""。
      - sort / limit / year / country：用同句、历史无关的 _extract_params 兜底。
    """
    name = tc.get("name", "")
    args = dict(tc.get("args", {}) or {})
    if name != "find_movies":
        return tc
    remove_kw = ["不再局限", "不再限制", "所有电影", "不限", "去掉", "不限类型",
                 "全部电影", "不限题材", "不限地区", "不要局限", "别局限"]
    force_clear = any(k in message for k in remove_kw)
    det_genres = _detect_genres_from_text(message)  # 只看当前句，不读历史
    if force_clear:
        args["genre"] = ""
    elif det_genres:
        args["genre"] = ",".join(det_genres)
    else:
        args["genre"] = ""  # 关键：当前句没提类型 → 不继承历史类型
    params = _extract_params(message)  # 同句抽取，历史无关
    if params.get("sort"):
        args["sort"] = params["sort"]
    if params.get("limit"):
        args["limit"] = params["limit"]
    if params.get("year"):
        args["year"] = params["year"]
    if params.get("country"):
        args["country"] = params["country"]
    tc["args"] = args
    return tc


def _collect_raw_with_label(tc: dict, result: str) -> str:
    """给工具真实结果按维度打上【排序：xxx】分组标记，供响应模板分别、逐一介绍。"""
    if tc.get("name") == "find_movies":
        sort = (tc.get("args") or {}).get("sort", "")
        label = _SORT_LABEL.get(sort, "默认") if sort else "评分最高（默认）"
        return f"【排序：{label}】\n{result}"
    return result


def _agent_run_with_guardrail(message: str, history, tr=None) -> str | None:
    """让模型用原生 function calling 自主决定工具调用（autonomous agent），代码护栏兜底。

    返回最终自然语言答案；若护栏重试耗尽仍没调到任何工具（模型持续偷懒），返回 None，
    由调用方退回确定性路由兜底。

    注意：为了让弱模型（glm-4-flash）在「自主」时不丢三落四，我们把两个【确定性安全网】
    以提示形式注入对话——它们不改变模型的自主决策权，只是把"原话里有哪些类型词 / 是否含已知片名"
    这类模型容易忽略的事实明确喂给它，避免重蹈「冒险题材的动画只传冒险」「把《你的名字》当科幻搜」的覆辙。
    """
    hint_msgs = _build_hint_messages(message)

    llm = _make_llm().bind_tools(TOOLS)
    base = [SystemMessage(content=system_prompt())]
    if hint_msgs:
        base.extend(hint_msgs)
    messages = base + list(history) + [HumanMessage(content=message)]
    tool_ever_called = False
    combined_raw: list[str] = []
    for _ in range(_MAX_AGENT_STEPS):
        resp = llm.invoke(messages)
        if getattr(resp, "tool_calls", None):
            # 模型决定调工具 → 代码执行（先用确定性逻辑修正参数，去历史污染），把结果喂回
            tool_ever_called = True
            messages.append(resp)
            for tc in resp.tool_calls:
                tc = _sanitize_tool_call(message, tc)
                result = _safe_invoke_tool(tc, tr)
                combined_raw.append(_collect_raw_with_label(tc, result))
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            continue
        # 模型给出了一段文本回答（没有继续调工具）
        if not tool_ever_called:
            # 护栏：涉及电影却没调工具 → 多半是凭记忆瞎答，强制重试（重试时再强调一遍提示）
            if tr is not None:
                tr.used_guardrail = True  # 埋点：本轮回退到护栏强制重试
            messages.append(resp)
            retry = _GUARDRAIL_MSG
            if hint_msgs:
                retry += "\n" + " ".join(m.content for m in hint_msgs)
            messages.append(SystemMessage(content=retry))
            continue
        return resp.content or ""
    if tool_ever_called:
        # 最终答案用确定性模板把【工具真实结果】自然复述（不带工具绑定 → 不会泄工具语法、零幻觉）
        return _respond_result(message, "\n\n".join(combined_raw), history)
    # 步数耗尽仍未调到工具 → 交给上层退回确定性路由
    return None


def _agent_run_with_guardrail_stream(message: str, history, tr=None):
    """_agent_run_with_guardrail 的流式版本：模型通过原生 function calling 自主调工具，
    但最终自然语言答案用 stream=True 逐 token 产出（yield 文本片段）。
    - 决策轮用 invoke（前端只关心最终作答，决策过程不展示）。
    - 工具参数先用确定性逻辑修正（去历史污染），杜绝「上轮动画 → 这轮仍带 genre=动画」。
    - 最终答案：把【工具真实结果】用确定性模板自然复述（不带工具绑定 → 不会泄工具语法、零幻觉），并流式产出。
    """
    hint_msgs = _build_hint_messages(message)
    llm = _make_llm().bind_tools(TOOLS)
    messages = [SystemMessage(content=system_prompt())]
    if hint_msgs:
        messages.extend(hint_msgs)
    messages = messages + list(history) + [HumanMessage(content=message)]
    tool_ever_called = False
    combined_raw: list[str] = []
    for _ in range(_MAX_AGENT_STEPS):
        resp = llm.invoke(messages)
        if getattr(resp, "tool_calls", None):
            tool_ever_called = True
            messages.append(resp)
            for tc in resp.tool_calls:
                tc = _sanitize_tool_call(message, tc)
                result = _safe_invoke_tool(tc, tr)
                combined_raw.append(_collect_raw_with_label(tc, result))
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            continue
        if not tool_ever_called:
            # 护栏：涉及电影却没调工具 → 强制重试（重试时再强调一遍提示）
            if tr is not None:
                tr.used_guardrail = True  # 埋点：本轮回退到护栏强制重试
            messages.append(resp)
            retry = _GUARDRAIL_MSG
            if hint_msgs:
                retry += "\n" + " ".join(m.content for m in hint_msgs)
            messages.append(SystemMessage(content=retry))
            continue
        break
    if tool_ever_called:
        raw = "\n\n".join(combined_raw)
        yield from _respond_result_stream(message, raw, history)
        return
    return  # 步数耗尽：交由上层退回确定性路由


# 资料库「未找到」确定性短路：工具明确返回未找到时，直接给安全回复，
# 绝不交给 LLM 自由发挥——否则弱模型会用训练记忆编造（如把机器人名说错、年份说错）。
def _is_not_found(raw: str) -> bool:
    """工具结果以「未找到」开头，即资料库无匹配。"""
    return bool(raw) and raw.strip().startswith("未找到")


def _not_found_reply(name: str | None = None) -> str:
    """资料库未收录时的确定性回复（不调 LLM，零幻觉）。"""
    if name:
        return (
            f"抱歉，资料库里暂时还没有收录《{name}》这部电影（可能还没收录哦）。\n"
            f"如果你想看类似风格的电影，我可以帮你推荐；或者你再多给我一点线索"
            f"（比如年份、导演、主演），我帮你再查查～"
        )
    return (
        "抱歉，资料库里暂时没有符合条件的电影（可能还没收录）。\n"
        "你可以换个筛选条件，或者告诉我你想看的风格，我帮你推荐类似的～"
    )


def _deterministic_movie_path(message: str, history, tr=None) -> str:
    """电影意图【主路径】：用我们已验证可靠的「代码抽参 + 代码调工具 + 自然语言」路径（仍零幻觉）。
    代码直接调工具（不再依赖弱模型会不会 function calling），100% 可靠、只多 1 次 LLM 润色。
    """
    forced_title = _find_title_in_text(message)
    if forced_title:
        # 从用户原话确定性认出已知片名 → 强制按该电影查库，绕开弱模型把「你的名字」当问助手名的偏见
        tool = _tool_by_name("get_movie_info_by_name")
        if tr is not None:
            tr.tool_calls.append({"name": "get_movie_info_by_name", "args": {"name": forced_title}})
        raw = str(tool.invoke({"name": forced_title}))
    else:
        params = _extract_params_rule(message)
        if params.get("defer"):
            return _respond_defer(message, history)
        raw = _route_tool(params, message, tr)
    if _is_not_found(raw):
        # 资料库明确未收录 → 确定性回复，绝不交给 LLM 自由发挥（防弱模型凭记忆编造）
        return _not_found_reply(forced_title)
    return _respond_result(message, raw, history)


def _run_deterministic_stream(message: str, history, tr=None):
    """_deterministic_movie_path 的流式版：确定性工具调用（同步）照常执行，
    最终自然语言用 stream=True 逐 token 产出。是电影意图的【默认流式路径】。

    流程：规则抽参（0 次 LLM）→ 代码调工具（真实 DB 数据）→ 模板复述（1 次 LLM 润色）。
    相比自主 agent 循环（最坏 3 轮 × LLM + 兜底 2 次 LLM = 5 次调用），
    这里固定 1 次 LLM 调用，延迟从 ~50s 降到 ~10s，且答案 100% 基于工具真实数据。
    """
    forced_title = _find_title_in_text(message)
    if forced_title:
        tool = _tool_by_name("get_movie_info_by_name")
        if tr is not None:
            tr.tool_calls.append({"name": "get_movie_info_by_name", "args": {"name": forced_title}})
        raw = str(tool.invoke({"name": forced_title}))
        if _is_not_found(raw):
            # 资料库明确未收录 → 确定性回复，绝不交给 LLM 自由发挥（防弱模型凭记忆编造）
            yield _not_found_reply(forced_title)
            return
        yield from _respond_result_stream(message, raw, history)
        return
    params = _extract_params_rule(message)
    if params.get("defer"):
        # 用户说"稍后再给要求 / 先别急"——不查库，自然回应等他开口
        prompt = (_DEFER_TEMPLATE.replace("{history}", _fmt_history(history))
                                      .replace("{question}", message))
        for t in _llm_reply_stream(prompt, [], message):
            yield t
        return
    raw = _route_tool(params, message, tr)
    if _is_not_found(raw):
        # 资料库明确未收录 → 确定性回复，绝不交给 LLM 自由发挥
        yield _not_found_reply(None)
        return
    yield from _respond_result_stream(message, raw, history)


def _respond_result_stream(message: str, raw_result: str, history):
    """_respond_result 的流式版：用 stream=True 逐 token 产出最终自然语言答案。"""
    prompt = (_RESPOND_TEMPLATE.replace("{history}", _fmt_history(history))
                                .replace("{question}", message)
                                .replace("{result}", raw_result))
    yield from _llm_reply_stream(prompt, [], message)


def _deterministic_movie_path_stream(message: str, history, tr=None):
    """_deterministic_movie_path 的流式版：确定性工具调用（同步）照常执行，
    但最终自然语言用 stream=True 逐 token 产出，避免兜底时「一次性全吐」。

    作为「主路径（_run_deterministic_stream）也空手而归」时的最后兜底（罕见，如工具全失败）。
    """
    forced_title = _find_title_in_text(message)
    if forced_title:
        tool = _tool_by_name("get_movie_info_by_name")
        if tr is not None:
            tr.tool_calls.append({"name": "get_movie_info_by_name", "args": {"name": forced_title}})
        raw = str(tool.invoke({"name": forced_title}))
    else:
        params = _extract_params(message)
        if params.get("defer"):
            return  # defer 在流式主流程已有独立分支处理，不会走到这
        raw = _route_tool(params, message, tr)
    if _is_not_found(raw):
        # 资料库明确未收录 → 确定性回复，绝不交给 LLM 自由发挥
        yield _not_found_reply(forced_title)
        return
    yield from _respond_result_stream(message, raw, history)


def chat(message: str, session_id: str = "default") -> str:
    """运行一次对话，返回助手文本（C 端 /api/agent/chat 用）。"""
    answer, _meta = _run_chat(message, session_id)
    return answer


def chat_with_meta(message: str, session_id: str = "default") -> tuple[str, dict]:
    """同 chat，但额外返回本次对话的可观测元数据（管理端 /api/admin/chat 用）。

    meta = {
        "intent": str, "tool_calls": list[dict], "used_guardrail": bool,
        "cache_hit": bool, "latency_ms": int,
    }
    """
    return _run_chat(message, session_id)


def _run_chat(message: str, session_id: str = "default") -> tuple[str, dict]:
    """对话主流程（统一入口）：返回 (answer, meta)，并把埋点落库到 agent_traces。

    混合架构（guardrailed agent）：
    - chat / identity：确定性处理（问候/感谢直接聊；身份问题固定自介），不调工具。
    - 电影相关：
        · 用户说「稍后再给要求 / 先别急」(defer) → 不查库、自然回应等他开口（听懂人话）。
        · 否则进入【代码驱动确定性路由】：规则抽参 + 代码调工具(_route_tool) + 模板复述，
          100% 可靠、零幻觉、仅 1 次 LLM 润色。
        · 仅当确定性路径空手而归（罕见，如工具全失败）→ 退回自主 function-calling agent 作最后兜底。
      这样既有真实 agent / 工具链（简历含金量），又把可靠性交给确定性代码。
    """
    t0 = time.time()
    tr = _Trace()
    tr.mode = "autonomous" if is_autonomous() else "deterministic"  # 提前确定，供缓存键与埋点使用
    trace_token = _TRACE_CTX.set(tr)
    uq_token = _USER_QUERY_CTX.set(message)  # 供 find_movies 做「类型词自动抽取」兜底
    try:
        # —— 响应缓存：同问题短期复问直接复用，命中即标记 cache_hit ——
        # 缓存键带 mode，确保「同问题切到另一种模式」会重新跑（否则对比演示会命中旧缓存）。
        cache_key = f"{tr.mode}|{message.strip().lower()}"
        with _RESP_CACHE_LOCK:
            cached = _RESP_CACHE.get(cache_key)
        if cached is not None:
            tr.cache_hit = True
            latency = int((time.time() - t0) * 1000)
            _record_trace(session_id, message, "cache_hit", tr, cached, latency)
            return cached, {
                "intent": "cache_hit", "tool_calls": [], "used_guardrail": False,
                "cache_hit": True, "latency_ms": latency,
            }

        intent = _classify_intent(message)
        store = _get_history(session_id)
        history = list(store.messages)

        if intent == "chat":
            # 纯问候（你好/在吗/hi…）确定性返回，省一次 LLM 往返；含实质内容的闲聊仍走 LLM
            answer = _GREET_REPLY if _is_pure_greeting(message) else _respond_chat(message, history)
        elif intent == "identity":
            # 身份类问题：固定自介，确定性返回（弱模型时好时坏，故不走 LLM）
            answer = _respond_identity(message, history)
        elif _is_defer(message):
            # 用户说"稍后再给要求 / 先别急"——不查库，自然回应等他开口
            answer = _respond_defer(message, history)
        else:
            # 电影相关：
            # - 规则路由模式（默认）：代码抽参 + 代码调工具 + 模板复述，可靠、零幻觉。
            # - 自主 Agent 模式（开关打开）：以 LLM 自主 function calling 为主路径，
            #   真实展示「模型自己选工具 + 护栏强制先查库」；确定性路由作兜底。
            if is_autonomous():
                tr.mode = "autonomous"
                answer = _agent_run_with_guardrail(message, history, tr) or ""
                if not answer:
                    # 自主路径空手而归（罕见，如弱模型持续偷懒）→ 退回确定性路由兜底
                    answer = _deterministic_movie_path(message, history, tr)
            else:
                tr.mode = "deterministic"
                answer = _deterministic_movie_path(message, history, tr)
                if not answer:
                    # 确定性路径空手而归（罕见，如工具全失败）→ 退回自主 agent 作最后尝试
                    answer = _agent_run_with_guardrail(message, history, tr) or ""

        store.add_user_message(message)
        store.add_ai_message(answer)
        _extract_prefs(message)

        latency = int((time.time() - t0) * 1000)
        meta = {
            "intent": intent,
            "tool_calls": tr.tool_calls,
            "used_guardrail": tr.used_guardrail,
            "cache_hit": False,
            "latency_ms": latency,
            "mode": tr.mode,
        }
        _record_trace(session_id, message, intent, tr, answer, latency)
        with _RESP_CACHE_LOCK:
            _RESP_CACHE[cache_key] = answer
        return answer, meta
    finally:
        _safe_reset(uq_token)
        _safe_reset(trace_token)


def chat_stream(message: str, session_id: str = "default"):
    """流式对话入口（C 端 /api/agent/chat/stream 用）。

    yield 的是 dict：
      {"type": "token", "text": "<片段>"}  —— 逐片文本
      {"type": "done",  "meta":  {...}}     —— 结束 + 可观测元数据（意图/工具链/护栏/缓存/耗时）
    由 API 层负责序列化成 SSE（token→data，done→event:done）。
    """
    return _run_chat_stream(message, session_id)


def _run_chat_stream(message: str, session_id: str = "default"):
    """_run_chat 的流式版：逻辑一致，但最终答案逐 token 产出。"""
    t0 = time.time()
    tr = _Trace()
    tr.mode = "autonomous" if is_autonomous() else "deterministic"  # 提前确定，供缓存键与埋点使用
    trace_token = _TRACE_CTX.set(tr)
    uq_token = _USER_QUERY_CTX.set(message)  # 供 find_movies 做「类型词自动抽取」兜底
    try:
        # —— 响应缓存：同问题短期复问直接复用，命中即标记 cache_hit ——
        # 缓存键带 mode，确保「同问题切到另一种模式」会重新跑（否则对比演示会命中旧缓存）。
        cache_key = f"{tr.mode}|{message.strip().lower()}"
        with _RESP_CACHE_LOCK:
            cached = _RESP_CACHE.get(cache_key)
        if cached is not None:
            tr.cache_hit = True
            latency = int((time.time() - t0) * 1000)
            _record_trace(session_id, message, "cache_hit", tr, cached, latency)
            yield {"type": "token", "text": cached}
            yield {"type": "done", "meta": {
                "intent": "cache_hit", "tool_calls": [], "used_guardrail": False,
                "cache_hit": True, "latency_ms": latency,
            }}
            return

        intent = _classify_intent(message)
        store = _get_history(session_id)
        history = list(store.messages)
        full = ""

        if intent == "chat":
            # 纯问候确定性返回（不调 LLM，~0ms）；其余闲聊照常流式 LLM
            if _is_pure_greeting(message):
                full = _GREET_REPLY
                yield {"type": "token", "text": full}
            else:
                for t in _llm_reply_stream(system_prompt(), history, message):
                    full += t
                    yield {"type": "token", "text": t}
        elif intent == "identity":
            # 身份类问题：固定自介，确定性返回（不交给弱模型自由发挥）
            full = _respond_identity(message, history)
            yield {"type": "token", "text": full}
        elif _is_defer(message):
            # 用户说"稍后再给要求 / 先别急"——不查库，自然回应等他开口
            prompt = (_DEFER_TEMPLATE.replace("{history}", _fmt_history(history))
                                          .replace("{question}", message))
            for t in _llm_reply_stream(prompt, [], message):
                full += t
                yield {"type": "token", "text": t}
        else:
            if is_autonomous():
                # 自主 Agent 模式：以 LLM 自主 function calling 为主路径，
                # 真实展示「模型自己选工具 + 护栏强制先查库」；确定性路由作兜底。
                tr.mode = "autonomous"
                yield {"type": "status", "text": "自主 Agent 模式：模型正在自主选工具…"}
                for t in _agent_run_with_guardrail_stream(message, history, tr):
                    full += t
                    yield {"type": "token", "text": t}
                if not full:
                    # 自主路径空手而归 → 退回确定性路由兜底
                    for t in _run_deterministic_stream(message, history, tr):
                        full += t
                        yield {"type": "token", "text": t}
            else:
                # 规则路由模式（默认）：代码驱动确定性路由，仅 1 次 LLM 润色，可靠且快。
                tr.mode = "deterministic"
                yield {"type": "status", "text": "正在检索电影库…"}
                for t in _run_deterministic_stream(message, history, tr):
                    full += t
                    yield {"type": "token", "text": t}
                if not full:
                    # 确定性路径空手而归（罕见，如工具全失败）→ 退回自主 agent 作最后尝试
                    for t in _agent_run_with_guardrail_stream(message, history, tr):
                        full += t
                        yield {"type": "token", "text": t}
                    if not full:
                        # 仍为空 → 再退回确定性兜底
                        for t in _deterministic_movie_path_stream(message, history, tr):
                            full += t
                            yield {"type": "token", "text": t}

        store.add_user_message(message)
        store.add_ai_message(full)
        _extract_prefs(message)

        latency = int((time.time() - t0) * 1000)
        meta = {
            "intent": intent,
            "tool_calls": tr.tool_calls,
            "used_guardrail": tr.used_guardrail,
            "cache_hit": False,
            "latency_ms": latency,
            "mode": tr.mode,
        }
        _record_trace(session_id, message, intent, tr, full, latency)
        with _RESP_CACHE_LOCK:
            _RESP_CACHE[cache_key] = full
        yield {"type": "done", "meta": meta}
    finally:
        _safe_reset(uq_token)
        _safe_reset(trace_token)
