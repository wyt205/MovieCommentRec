"""Agent 装配：用 LangChain 把「大模型 + 工具 + 会话记忆」串成一个【有护栏的自主智能体（guardrailed agent）】。

核心思路（hybrid 架构）：
- 闲聊 / 身份 / 暂缓(defer)：用【确定性代码】直接处理，不依赖弱模型的"自觉"，可靠且零幻觉。
- 电影相关：让模型通过【原生 function calling】(bind_tools) 自主决定调哪个工具、并可多步串联
  （例如「分别推评分最高+热度最高的电影」会自动调两次 find_movies）。
- 【代码护栏】保证可靠性：若模型某轮"涉及电影却没调工具"（弱模型常见的凭记忆瞎答惯性），
  强制重试；若护栏重试耗尽仍未调工具，退回我们已验证可靠的【确定性路由】兜底（仍零幻觉）。

这样既保留真实 agent 的自主决策（简历含金量），又有可靠性兜底——绝不会因为模型偷懒而幻觉。

未配置 llm_api_key 时，_make_llm() 会抛 RuntimeError，由路由层转成 503 友好提示，
因此「模型待定」阶段后端也能照常启动、其它功能不受影响。
"""

import json
import re
import time
import threading
from contextvars import ContextVar

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.ai.tools import TOOLS, _USER_QUERY_CTX, _detect_genres_from_text
from app.core.config import settings
from app import crud
from app.db.database import SessionLocal
from app.models import AgentTrace

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


# 记忆存储（内存，演示用；生产可换 Redis / 数据库持久化）
_HISTORY_STORE: dict[str, BaseChatMessageHistory] = {}

# —— 埋点层（管理端可观测性的地基）——
# 每次对话一个 trace 对象，挂在 contextvar 上，供 _safe_invoke_tool / 护栏 写入工具调用链，
# 对话结束统一落库到 agent_traces 表。管理端的「日志 / 缓存命中 / 护栏使用率」都读这张表。
_TRACE_CTX: ContextVar = ContextVar("agent_trace", default=None)


class _Trace:
    """单次对话的可观测数据收集器。"""

    def __init__(self):
        self.tool_calls: list[dict] = []
        self.used_guardrail: bool = False
        self.cache_hit: bool = False


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
    if session_id not in _HISTORY_STORE:
        from langchain_core.chat_history import InMemoryChatMessageHistory

        _HISTORY_STORE[session_id] = InMemoryChatMessageHistory()
    return _HISTORY_STORE[session_id]


def _make_llm():
    """构造大脑 LLM（参数提取 / 闲聊共用）。未配置 key 时抛 RuntimeError（由路由层兜底）。"""
    if not settings.llm_api_key:
        raise RuntimeError("未配置 LLM_API_KEY，Agent 未启用。请在 .env 中设置 llm_api_key。")
    return ChatOpenAI(
        model=settings.llm_model or "glm-4-flash",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        temperature=0.3,
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


# 排序维度 → 中文标签（用于向响应层说明每个分组是按什么排的）
_SORT_LABEL = {
    "rating": "评分最高",
    "popularity": "热度最高",
    "year": "年份",
    "release": "上映日期",
}


def _route_tool(params: dict, message: str) -> str:
    """根据结构化参数，在代码里直接调用对应工具，返回工具的真实文本结果。
    message 用于注入 contextvar（供 find_movies 做类型词自动抽取兜底）。

    支持「多排序维度」：sort 以逗号分隔多个维度时，分别调用 find_movies 并分块返回，
    从而能正确满足「分别推一个评分最高和一个热度最高」这类诉求（两者往往是不同电影，
    此前只查一次会把同一部电影包装成两个，造成「评分最高=热度最高」的离谱结果）。
    """
    token = _USER_QUERY_CTX.set(message)
    try:
        mode = params.get("mode", "find")
        if mode == "movie_info" and params.get("name"):
            tool = _tool_by_name("get_movie_info_by_name")
            return str(tool.invoke({"name": params["name"]}))
        if mode == "semantic":
            tool = _tool_by_name("semantic_search_movies")
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
            raw = str(tool.invoke({
                "genre": genre,
                "year": year,
                "country": country,
                "sort": s,
                "limit": per_limit,
            }))
            # 用户明确要 N 部、但资料库不足 N 部时，诚实告知（而非假装凑满）
            m = re.search(r"找到 (\d+) 部", raw)
            if m and requested_limit > 0:
                cnt = int(m.group(1))
                if 0 < cnt < requested_limit:
                    raw += (
                        f"\n（注：资料库里符合条件的电影目前共 {cnt} 部，已全部列出，"
                        f"未能凑满你想要的 {requested_limit} 部。）"
                    )
            label = _SORT_LABEL.get(s, "默认") if s else "评分最高（默认）"
            blocks.append(f"【排序：{label}】\n{raw}")
        return "\n\n".join(blocks)
    finally:
        _USER_QUERY_CTX.reset(token)


# 把工具的「真实结果」用自然、会"听人话"的中文讲给用户（事实仍 100% 来自工具，零幻觉）
_RESPOND_TEMPLATE = """你是智影影评网站的智能助手，正在和用户多轮聊电影。

【铁律 · 只能基于工具结果说话】
- 你只能依据下面【工具结果】里给出的信息来谈论电影。禁止编造任何电影名、评分、年份、类型、地区、简介。
- 若【工具结果】是「未找到 / 没有符合条件的电影」，就如实、友好地告诉用户资料库里暂时没有这部电影（可能还没收录），并可以友好地问一句「你想看什么类型的？我帮你推荐类似的」。
- 若【工具结果】是一组电影，就用自然、亲切的口吻介绍这批电影（可加一句引导语，例如「给你挑了几部符合要求的电影：」），不要生硬复述原始格式；保留片名、评分、类型等关键信息即可。
- 若【工具结果】是一段「单部电影资料」（以《片名》开头、含导演/演员/评分/简介等字段），你必须介绍【那部电影】本身。特别注意：即使用户的话里出现了「你的名字」这类措辞，只要工具结果是一份电影资料，就说明用户在问那部【电影】（例如《你的名字。》），你只能介绍该电影；**绝不要**把它理解成在问「助手你叫什么名字」，也**绝不要**输出任何关于「助手身份 / 助手名字 / 我是谁」的内容。
- 若【工具结果】包含多个以「【排序：xxx】」开头的分组，说明是按不同维度（如评分最高 / 热度最高）分别查到的电影。请【分别、逐一】介绍每个分组，并明确点出该组是按什么维度排的（如「这是评分最高的一部」「这是热度最高的一部」）；不同分组往往是不同的电影，绝对不要把它们混为一谈、也不要只介绍其中一个而漏掉其它分组。
- 【工具结果】里的「【排序：xxx】」只是内部分组标记，组织答案时不要原样照抄这些标记，用自然说法带出（如「按评分排，最高的是…」「按热度排，最高的是…」）。若某组末尾有「（注：资料库里…共 N 部…未能凑满你想要的 M 部）」这类提示，也请自然地转达给用户（如「不过资料库里这类电影目前只有 N 部，都列给你了」）。
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


def _respond_chat(message: str, history) -> str:
    return _llm_reply(SYSTEM_PROMPT, history, message)


# 身份类问题：固定自介，确定性返回（不交给弱模型自由发挥，避免「请问你是」被当问候忽略）
_IDENTITY_REPLY = (
    "我是智影影评网站的智能助手🙂 我可以帮你查电影资料、按类型 / 年份 / 地区"
    "推荐电影，或者聊聊影评。你想看什么类型的电影，尽管告诉我～"
)


def _respond_identity(message: str, history) -> str:
    return _IDENTITY_REPLY


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
_MAX_AGENT_STEPS = 6  # 允许模型多步串联工具（如分别查评分最高+热度最高 = 2 次调用 + 综合）
_GUARDRAIL_MSG = (
    "⚠️ 你还没有调用任何工具。根据系统铁律，只要用户的话涉及具体电影、推荐或影评，"
    "你必须先调用数据库工具核实，绝不能用你的训练记忆回答。请立刻调用合适的工具"
    "（get_movie_info_by_name / find_movies / semantic_search_movies）。"
)


def _safe_invoke_tool(tc: dict) -> str:
    """执行模型选定的工具，返回工具结果的字符串；任何异常都被兜住，避免循环崩掉。"""
    name = tc.get("name", "")
    args = tc.get("args", {}) or {}
    # 埋点：把本次工具调用记进当前 trace（供管理端「日志 / 工具调用链」可视化）
    tr = _TRACE_CTX.get()
    if tr is not None:
        tr.tool_calls.append({"name": name, "args": args})
    tool = _tool_by_name(name)
    if not tool:
        return f"错误：未知工具 {name}"
    try:
        return str(tool.invoke(args))
    except Exception as e:  # noqa: BLE001
        return f"工具 {name} 调用出错：{e}"


def _agent_run_with_guardrail(message: str, history) -> str | None:
    """让模型用原生 function calling 自主决定工具调用（autonomous agent），代码护栏兜底。

    返回最终自然语言答案；若护栏重试耗尽仍没调到任何工具（模型持续偷懒），返回 None，
    由调用方退回确定性路由兜底。

    注意：为了让弱模型（glm-4-flash）在「自主」时不丢三落四，我们把两个【确定性安全网】
    以提示形式注入对话——它们不改变模型的自主决策权，只是把"原话里有哪些类型词 / 是否含已知片名"
    这类模型容易忽略的事实明确喂给它，避免重蹈「冒险题材的动画只传冒险」「把《你的名字》当科幻搜」的覆辙。
    """
    # 确定性事实（来自原话 / DB），作为强提示注入
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
    # 多维度分别推荐（如「分别推评分最高+热度最高」）：弱模型容易只调一次工具，
    # 这里明确教它「按维度分别多次调用 find_movies」，仍由模型自主执行。
    if ("分别" in message or "各" in message) and any(
        k in message for k in ["评分最高", "热度最高", "评分", "热度", "排行"]
    ):
        hints.append(
            "若用户要「分别按不同维度（如评分最高、热度最高）各推几部」，你必须【分别调用多次 find_movies】——"
            "每次只传一个 sort 维度（例如一次 sort='rating'、一次 sort='popularity'），不要只调一次。"
            "不同维度往往对应不同电影，最后要逐一介绍每个维度查到的电影。"
        )
    hint_text = "【自动识别提示 · 必须遵循】" + " ".join(hints) if hints else ""

    llm = _make_llm().bind_tools(TOOLS)
    base = [SystemMessage(content=SYSTEM_PROMPT)]
    if hint_text:
        base.append(SystemMessage(content=hint_text))
    messages = base + list(history) + [HumanMessage(content=message)]
    tool_ever_called = False
    for _ in range(_MAX_AGENT_STEPS):
        resp = llm.invoke(messages)
        if getattr(resp, "tool_calls", None):
            # 模型决定调工具 → 代码执行，把结果喂回，让模型接着决策（可多步）
            tool_ever_called = True
            messages.append(resp)
            for tc in resp.tool_calls:
                result = _safe_invoke_tool(tc)
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            continue
        # 模型给出了一段文本回答（没有继续调工具）
        if not tool_ever_called:
            # 护栏：涉及电影却没调工具 → 多半是凭记忆瞎答，强制重试（重试时再强调一遍提示）
            tr = _TRACE_CTX.get()
            if tr is not None:
                tr.used_guardrail = True  # 埋点：本轮回退到护栏强制重试
            messages.append(resp)
            retry = _GUARDRAIL_MSG
            if hint_text:
                retry += "\n" + hint_text
            messages.append(SystemMessage(content=retry))
            continue
        return resp.content or ""
    # 步数耗尽仍未调到工具 → 交给上层退回确定性路由
    return None


def _deterministic_movie_path(message: str, history) -> str:
    """护栏失败时的兜底：用我们已验证可靠的「代码抽参 + 代码调工具 + 自然语言」路径（仍零幻觉）。"""
    forced_title = _find_title_in_text(message)
    if forced_title:
        # 从用户原话确定性认出已知片名 → 强制按该电影查库，绕开弱模型把「你的名字」当问助手名的偏见
        tool = _tool_by_name("get_movie_info_by_name")
        raw = str(tool.invoke({"name": forced_title}))
    else:
        params = _extract_params(message)
        if params.get("defer"):
            return _respond_defer(message, history)
        raw = _route_tool(params, message)
    return _respond_result(message, raw, history)


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
        · 否则进入【有护栏的工具调用循环】：模型通过原生 function calling 自主决定调哪个工具、
          并可多步串联（如分别查评分最高+热度最高）；代码护栏保证「涉及电影必须先调工具，
          禁止凭记忆答」——若模型某轮没调工具就强制重试；若护栏重试耗尽仍未调工具，
          则退回我们已验证可靠的【确定性路由】兜底（仍零幻觉）。
      这样既有真实 agent 的自主决策（简历含金量），又有可靠性兜底。
    """
    t0 = time.time()
    tr = _Trace()
    trace_token = _TRACE_CTX.set(tr)
    uq_token = _USER_QUERY_CTX.set(message)  # 供 find_movies 做「类型词自动抽取」兜底
    try:
        # —— 响应缓存：同问题短期复问直接复用，命中即标记 cache_hit ——
        cache_key = message.strip().lower()
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
            # 闲聊/感谢：纯对话，不调工具
            answer = _respond_chat(message, history)
        elif intent == "identity":
            # 身份类问题：固定自介，确定性返回（弱模型时好时坏，故不走 LLM）
            answer = _respond_identity(message, history)
        elif _is_defer(message):
            # 用户说"稍后再给要求 / 先别急"——不查库，自然回应等他开口
            answer = _respond_defer(message, history)
        else:
            # 电影相关：先让模型自主用工具（autonomous agent），护栏兜底
            answer = _agent_run_with_guardrail(message, history)
            if answer is None:
                # 护栏重试仍没调到工具 → 退回已验证可靠的确定性路由（仍零幻觉）
                answer = _deterministic_movie_path(message, history)

        store.add_user_message(message)
        store.add_ai_message(answer)

        latency = int((time.time() - t0) * 1000)
        meta = {
            "intent": intent,
            "tool_calls": tr.tool_calls,
            "used_guardrail": tr.used_guardrail,
            "cache_hit": False,
            "latency_ms": latency,
        }
        _record_trace(session_id, message, intent, tr, answer, latency)
        with _RESP_CACHE_LOCK:
            _RESP_CACHE[cache_key] = answer
        return answer, meta
    finally:
        _USER_QUERY_CTX.reset(uq_token)
        _TRACE_CTX.reset(trace_token)
