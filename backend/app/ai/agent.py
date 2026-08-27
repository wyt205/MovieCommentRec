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
import os
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

# 评测严谨化：允许通过环境变量固定 LLM temperature。
# 默认 0.3（给正常对话留一点自然度）；run_eval.py 跑评测时会设 AGENT_LLM_TEMPERATURE=0，
# 关闭随机性 → 输出确定性、可复现，评测分数稳定、两次跑可直接对比（否则免费模型非确定性会让分数抖动）。
LLM_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.3"))

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

【资料库没有这部电影时 · 诚实 + 可补充公开知识】
- 当 get_movie_info_by_name(具体片名) 返回「未找到」时，说明资料库里暂时没有这部电影（可能还没收录）。
- **针对用户点名问的「某部具体电影」**：你【可以】基于公开知识简要介绍这部电影（导演、年份、类型、剧情、口碑）来帮助用户；但**必须明确声明**「资料库暂未收录该片，以下内容基于公开知识、仅供参考」，绝不可假称资料库里有这些数据，也绝不可编造不确定的具体评分/票房数字。
- **针对「按条件筛选」类**（如「悬疑片有哪些」）返回「未找到符合条件」：直接如实告诉用户资料库暂无此类电影即可，不要凭记忆编造片单。
- 无论哪种情况，**绝不要反过来追问这部电影本身的年份/导演/主演等细节**（你本来也不知道），也**不要**再去调用 semantic_search_movies 试图「确认」它是否存在。

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
            "导演", "主演", "剧情", "简介", "上映", "哪年", "谁演",
            "听过", "看过", "听说过", "听说"]
_RECOMMEND_KW = ["推荐", "找一些", "找几部", "有哪些", "有没有", "想看", "类似",
                 "什么类型", "题材", "排行", "评分最高", "热度最高", "帮我找",
                 "给我找", "来一部", "来几部", "按", "有啥", "有什么",
                 "帮我挑", "帮我选", "挑选", "挑一部", "选一部", "挑几部", "选几部"]
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
    顺序（关键）：identity → name → recommend → 纯问候/感谢 → 兜底 movie_related。

    【致命坑 · 已修】纯问候/感谢检查必须放在【最后】！原先它排在最前会短路返回 chat，
    导致「你好啊，推荐一部爱情片」「谢谢，帮我找部动作片」这类带问候前缀的推荐诉求
    被误判成闲聊，退化成纯 LLM 记忆作答（凭训练记忆编造库里没有的电影，如《重庆森林》）。
    只有排除了所有电影/身份信号后，才安全当作闲聊。
    - identity 仍放 name 之前：避免「你叫什么名字」里的"叫"被 _NAME_KW 误判成查片名；
      同时用 _MOVIE_SIGNAL_KW 抑制，保证「你知道《你是谁》吗」仍走查库而非身份闲聊。
    """
    m = message.lower()
    # 1) 身份问题（含身份词且不含电影信号）—— 放最前，避免「叫」被当查片名
    if any(k in m for k in _IDENTITY_KW) and not any(k in m for k in _MOVIE_SIGNAL_KW):
        return "identity"
    # 2) 具体电影名（你知道X吗 / 讲X的导演）→ 查库
    if any(k in m for k in _NAME_KW):
        return "movie_name"
    # 3) 推荐/找片诉求（含「推荐/想看/找/有什么」等）→ 走工具路由，务必先查库
    if any(k in m for k in _RECOMMEND_KW):
        return "recommend"
    # 3.5) 续轮推荐（换一个/还有吗/再来一部/再推荐…）：即使不含电影信号词，
    #      也是明确的「再要一部」诉求（要继承上一轮类型），绝不能当闲聊——
    #      否则会退化成纯 LLM 凭训练记忆瞎推库外电影（如「换一个」→ 编出《消失的她》）。
    #      注意放在 纯问候/感谢 之前，避免「好的，换一个吧」被「好的」短路成闲聊。
    if _is_followup(m):
        return "recommend"
    # 4) 上面都不是 → 才当作纯问候/感谢/闲聊（此时已排除所有电影/身份信号，安全）
    if any(k in m for k in _THANKS_KW) or any(k in m for k in _GREET_KW):
        return "chat"
    # 5) 兜底：含电影信号（电影/类型/影评/评分/看过…）→ 走电影分支；
    #    完全不含任何电影信号（如「今天有点累」「讲个笑话」）→ 纯闲聊，交 LLM 自然回应，
    #    避免把日常闲聊误判成「找电影」而推一屏幕评分最高电影（用户实测反馈）。
    if any(k in m for k in _MOVIE_SIGNAL_KW):
        return "movie_related"
    return "chat"


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

# 当前会话 id（供「会话级已推荐去重」使用）：在 _run_chat / _run_chat_stream 入口注入，
# _route_tool 读它来记住「本会话已经给用户推荐过哪些电影」，从而避免「再推荐/换一个/随机」
# 反复命中同一部。用 contextvar 而非改一堆中间函数签名，零侵入。
_SESSION_ID_CTX: ContextVar = ContextVar("session_id", default="default")


def _safe_reset(token):
    """安全重置 ContextVar；吞掉跨上下文（如测试客户端 worker 线程、某些 ASGI 服务器
    把生成器挪到别的上下文执行）可能抛出的 ValueError，避免流式响应尾部误报 error 事件。"""
    if token is None:
        return
    try:
        token.var.reset(token)
    except (ValueError, RuntimeError):
        pass


# ---------------------- 会话级「已推荐去重」 ----------------------
# 用户明确吐槽过：随机推荐十次都推同一部、再推荐也还是同一部——本质是「没有任何记忆、
# 每次都从全池裸查」。这里用「会话维度 + 已推荐电影 id 集合」做去重：
#   · 查库前排除本会话已推荐过的 id；
#   · 查库后把本次结果里的 id 记进集合；
#   · 集合仅在「排除后查无结果（池被耗尽）」时清空一次，避免无限膨胀也无片可推。
_SESSION_RECOMMENDED: dict[str, set] = {}
_SESSION_RECOMMENDED_LOCK = threading.Lock()


def _recommended_exclude(sid: str) -> set:
    """返回本会话应排除的电影 id 集合（查库前调用）。匿名会话（default）不做去重。"""
    if not sid or sid == "default":
        return set()
    with _SESSION_RECOMMENDED_LOCK:
        return _SESSION_RECOMMENDED.get(sid, set()).copy()


def _record_recommended(sid: str, combined_text: str) -> None:
    """把本次推荐结果里出现的电影 id 记进本会话集合（查库后调用）。"""
    if not sid or sid == "default":
        return
    ids = set(int(x) for x in re.findall(r"id=(\d+)", combined_text or ""))
    if not ids:
        return
    with _SESSION_RECOMMENDED_LOCK:
        bucket = _SESSION_RECOMMENDED.setdefault(sid, set())
        bucket.update(ids)
        # 防御：单会话不可能真推几百部，超限说明异常（如 exclude 逻辑出 bug），重置以免堆积
        if len(bucket) > 200:
            _SESSION_RECOMMENDED[sid] = set()


def _clear_recommended(sid: str) -> None:
    """清空本会话记忆（池耗尽兜底用）。"""
    if not sid or sid == "default":
        return
    with _SESSION_RECOMMENDED_LOCK:
        _SESSION_RECOMMENDED.pop(sid, None)


# ---------------------- 会话级「上一轮 find 筛选条件」 ----------------------
# 用户反复吐槽「再推荐 / 换一个 / 还有别的」会脱离前文（如聊完爱情片再说「再推荐一部」，
# 却推回非爱情片）。根因：_extract_params_rule 是无状态的，只看当前句，抽不到 genre。
# 而 LLM 润色环节被「铁律」锁死只能照工具结果说，无法回补 genre——所以上下文继承
# 必须做在【检索层】。这里记住本会话上一轮 find 的 genre/country/year/sort，续轮未明说时继承。
_SESSION_LAST_FIND: dict[str, dict] = {}
_SESSION_LAST_FIND_LOCK = threading.Lock()

# 续轮推荐意图关键词：命中即视为「接着上一轮继续推荐」，可继承上一轮筛选条件。
_FOLLOWUP_KW = [
    "再推荐", "再推", "再给", "再整", "换一个", "换一部", "再来", "再来一个",
    "再来一部", "还有", "别的", "其他", "其它", "另外", "类似的", "另一",
    "重新推", "重新推荐", "继续推", "还来", "也来", "换换",
]


def _is_followup(m: str) -> bool:
    """当前句是否「续轮推荐」意图（再推荐 / 换一个 / 还有别的…）。"""
    return any(k in m for k in _FOLLOWUP_KW)


def _get_last_find(sid: str) -> dict:
    """取本会话上一轮 find 的筛选条件（查库前继承用）。匿名会话返回空。"""
    if not sid or sid == "default":
        return {}
    with _SESSION_LAST_FIND_LOCK:
        return dict(_SESSION_LAST_FIND.get(sid, {}))


def _set_last_find(sid: str, genre: str, country: str, year, sort: str,
                   mode: str = "find", query: str = "") -> None:
    """记录本会话本轮 find 的筛选条件（查库后调用，供续轮继承）。

    mode/query：记录上一轮是「语义检索」还是「按条件筛选」——续轮「再推荐/换一个」
    未明说条件时，语义轮继承语义 query（保持主题一致），条件轮继承 genre/country/year/sort。
    """
    if not sid or sid == "default":
        return
    with _SESSION_LAST_FIND_LOCK:
        _SESSION_LAST_FIND[sid] = {
            "mode": mode or "find",
            "genre": genre or "",
            "country": country or "",
            "year": year or 0,
            "sort": sort or "",
            "query": query or "",
        }


class _Trace:
    """单次对话的可观测数据收集器。"""

    def __init__(self):
        self.tool_calls: list[dict] = []
        self.used_guardrail: bool = False
        self.cache_hit: bool = False
        self.mode: str = "deterministic"  # deterministic=规则路由主路径；autonomous=LLM 自主 function calling 主路径


# 注：早期版本曾用进程内 _RESP_CACHE 缓存「同问题复问」的答案，但该设计会让
# 「重复提问/随机推荐」每次都返回同一份旧答案（用户明确反对），故已彻底移除。
# 现在每次对话都真实走工具查询；管理端 cache_hit 字段恒为 False（仅保留兼容，
# 指标会如实显示 0% 命中率，不会因无缓存而崩溃）。


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
        # 关键：必须开 streaming=True，否则 .stream() 不会发起真正的流式 HTTP 请求，
        # 而是先完整生成、再把整段答案作为一个 chunk 一次性 yield —— 表现就是
        # 「思考很久、然后一下子全吐出来」，看起来像流式失效。开启后 .stream() 才会
        # 逐 token 产出，配合前端 getReader() 实现真正的逐字流式。
        # （.invoke() 不受影响，仍走普通非流式请求。）
        streaming=True,
        # temperature 由 LLM_TEMPERATURE 常量控制（默认 0.3；评测时由 AGENT_LLM_TEMPERATURE 环境变量覆盖为 0）
        temperature=LLM_TEMPERATURE,
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


def _extract_unknown_movie_name(m: str) -> str | None:
    """从「问某部具体电影」的口语里抽取片名（库内未知也抽），覆盖常见点名句式，
    用于 _find_title_in_text 为空（库内无此片）时兜底，确保仍先走 get_movie_info_by_name 查库。
    仅返回 2~16 字、且非指代/疑问代词的候选，避免误伤「帮我挑一部好看的悬疑片」这类推荐句。

    【根因背景】此前 _extract_params_rule 只认《》和库内已知片名，认不出「你知道X吗」这种
    没书名号、且库里没有的片名，会退化成泛化 find_movies 返回 TOP 列表，弱模型随之凭训练
    记忆编造该片简介（超能陆战队 bug 即此）。本函数把这类点名句式也归一到 movie_info 模式。
    """
    _STOP = {"这部", "那部", "这个", "那个", "它", "电影", "那部电影", "这部电影", "什么", "哪部"}
    # 认知类动词（你知道/了解/认识/听过/看过/听说…），覆盖：
    #   「你(有没有)听说过X吗」「你听过X吗」「你看过X吗」「你知道X吗」「听说X」
    # 结尾 吗? 可选，兼容「你有没有听说过X」这种【没有吗】的口语。
    _PATS = [
        # 认知类动词（你知道/了解/认识/听过/看过/听说…），覆盖肯定式与「A-not-A 反复疑问」式：
        #   肯定：你知道X / 你了解X / 你认识X / 你听说过X / 你听过X / 你看过X / 你看过X
        #   A-not-A：你知不知道X / 你认不认识X / 你了不了解X（用「不」）
        #           你听没听说过X / 你听没听X / 你看没看X / 你看过没看X（用「没+过」）
        #           你不知道X / 你不认识X / 你没听说X（否定式，同样指向具体片名）
        # 关键：A-not-A 形态里动词首字后紧跟「不/没」而非动词本体（如「你知|不|知道」），
        #   不能只写「你知道」——否则「你知不知道火影忍者」在首字就匹配不到「知道」而整体失败、
        #   退化成泛化检索返回 TOP 列表（火影忍者 bug 根因）。结尾 吗? 可选。
        r"你(?:有没有)?(?:"
        r"知不知道|认不认识|了不了解|听没听说过|看没看过|看过没看|听没听|看没看|"
        r"不知道|不了解|不认识|没听说过|没听过|没看过|没听|没看|"
        r"听说过|听过|看过|知道|了解|认识|听说|听|看"
        r")[《]?([^《》，。？！\s]{2,16})[》]?吗?",
        r"听说[过]?[《]?([^《》，。？！\s]{2,16})[》]?",
        r"[《]?([^《》，。？！\s]{2,16})[》]?讲(?:的是什|的什|了什|什|的是|的|是|了)?[么吗嘛啥]",
        r"[《]?([^《》，。？！\s]{2,16})[》]?的(?:导演|演员|主角|剧情|简介|上映|评分)",
        r"[《]?([^《》，。？！\s]{2,16})[》]?好看吗",
        r"[《]?([^《》，。？！\s]{2,16})[》]?(?:怎么样|如何)",
        r"(?:介绍|说说|讲讲|聊一聊)[一]?[下下]?[《]?([^《》，。？！\s]{2,16})[》]?",
        r"[《]?([^《》，。？！\s]{2,16})[》]?是[一]?[部部]?什么电影",
    ]
    # 句末语气词 + 指代词，抓到片名后先剥掉（避免把「吗/的/这部」带进片名）
    _PARTICLES = "《》的这部那部那个这个吗呢吧啊呀哦嘛"
    # 片名后常跟的角色/属性/描述词，遇到即截断（防止贪心抓到「超能陆战队导演」「超能陆战队挺好看」）。
    # 注意：只用【多字角色词】与【绝不会开头片名的副词】(挺/很/特别/非常)；
    # 绝不加「超/还/最/真/也/又」等——它们会开头真实片名（超人/还珠格格/最后的武士/真爱），误杀。
    _TERMINATORS = ("导演", "演员", "主演", "剧情", "简介", "上映", "评分", "讲的", "好看",
                    "怎么样", "如何", "什么电影", "挺", "很", "特别", "非常")
    for p in _PATS:
        mt = re.search(p, m)
        if not mt:
            continue
        nm = mt.group(1).strip(_PARTICLES)
        for role in _TERMINATORS:
            if role in nm:
                nm = nm.split(role)[0]
        # 剥掉尾部的「指代 + 类别词」（如「奇幻大冒险那部动画片」→「奇幻大冒险」、
        # 「这片子」→「」），只留真正的片名候选；再交给 get_movie_info_by_name 模糊匹配。
        nm = re.sub(r"(?:这部|那部|这个|那个)?(?:动画片|动漫|电影|影片|片子|剧)$", "", nm).strip(_PARTICLES)
        # 过滤指代/疑问词（如「你知道这部电影叫什么吗」会误抓「这部电影叫什么」）
        if not nm or nm in _STOP or len(nm) < 2:
            continue
        if any(b in nm for b in ("什么", "哪", "叫", "咋", "谁", "怎么", "名字", "电影", "影片")):
            continue
        return nm
    return None


def _extract_title_fragment(m: str) -> str | None:
    """片名片段搜索：用户记不清完整片名、只记得「标题/片名/名字里有X」「片名带X」等。
    返回标题片段（如「冒险王」），供 search_movies 做模糊匹配。

    【为何必须在 genre 检测之前】「冒险王」里含类型词「冒险」，若先跑 _detect_genres_from_text
    会被当成 genre=冒险，进而退化成 find_movies(genre=冒险) 推评分最高的冒险片（瞎推 bug 根因）。
    故本函数在 _extract_params_rule 的 genre 步骤前执行，命中即走 name_search → search_movies。
    """
    # 捕获字符类刻意排除「的/这/那/有/哪/都/也/就/还/呢/吧/啊/吗/啥/怎/谁」等自然断词，
    # 让片段在遇到「的电影 / 这三个字 / 那个电影」等赘余前就停住，避免贪心吞掉整句。
    _FRAG_CH = r"[^，。？！\s，。？！、的这那有哪都也就还呢吧啊吗啥怎谁叫为是中上下放着]"
    _PATS = [
        # 显式标记「标题/片名/名字/电影名/剧名/片子/影片 + 里有/带/包含/是/叫」
        r"(?:标题|片名|名字|电影名|影片名|剧名|片子|影片)[里中上]?[的有带包含叫为]\s*(" + _FRAG_CH + r"{1,16})",
        # 「记得/忘了/只记得 + (标题/片名/名字) + 里有/带/包含/是/叫」
        r"(?:记得|忘了|不记得|记不清|只记得)[^，。？！]{0,12}?(?:标题|片名|名字|电影名)[里中上]?[的有带包含叫为]\s*(" + _FRAG_CH + r"{1,16})",
    ]
    for p in _PATS:
        mt = re.search(p, m)
        if not mt:
            continue
        frag = mt.group(1)
        # 剥掉末尾「这三个字 / 那几个字 / 几个字 / 个字」等量词语赘（用户常补「这三个字」）
        frag = re.sub(r"^(这|那|几|个)?(三|几|一)?个字$", "", frag)
        frag = frag.strip("的有带包含叫为是，。？！、 ")
        if frag and 1 <= len(frag) <= 16:
            return frag
    return None


# 参数提取提示词：让模型把用户问题转成结构化参数（模型只做"抽取"，不做"是否查库"的决策）
_EXTRACT_PROMPT = """你是一个参数提取器。根据用户的问题，提取查询电影数据库所需的参数，只返回一段 JSON（不要任何额外文字、不要 markdown 代码块）。

判定与字段：
- mode="movie_info"：用户明显在问「某部具体电影」本身（如「你知道X吗」「X讲什么」「X的导演/演员/剧情」「X是动画吗」）。填 name（尽量还原准确片名，如「超能陆战队」「指环王」）。
- mode="find"：用户按条件找/推荐电影（如「推荐动画」「评分最高的动作片」「2021日本电影」「冒险题材的动画」）。填：
    genre：类型，多个用逗号隔开（如 动画,冒险），没有则空字符串
    year：年份数字，没有则 0
    country：国家/地区中文（如 美国/日本），没有则空字符串
    sort：排序 rating(评分)/popularity(热度)/year(年份)/release(上映日期)/random(随机，用于「随机/随便推荐一部」)。**若用户要「分别按不同维度各推几部」（如「分别推一个评分最高和一个热度最高的电影」），把多个维度用英文逗号写在 sort 里（如 sort="rating,popularity"），并把 limit 设为每个维度想要的条数（「一个」→limit=1）。** 用户说「随机/随便推荐」时填 random。没有则空字符串
    limit：想要的条数，没有则 0
- mode="semantic"：用户按「剧情/主题/情感/氛围」描述想看的电影（如「讲时间循环」「结局治愈」「深夜一个人看」），没有具体片名也没有明确类型筛选。填 query（用户原意描述）。
- mode="chat"：纯闲聊/问候/感谢（如「你好」「谢谢」「好的」），无需查库。
- defer：布尔值。仅当用户明确表示「稍后再给具体要求 / 还没想好 / 先别急着推荐 / 等我说 / 我一会儿会给你要求 / 你先别急」时为 true；只要用户已经给出了可查询的具体条件（类型/年份/地区/片名/主题），就为 false。注意：用户说「推荐几个电影」但紧接着说「我一会儿会给你要求」，说明此刻还不该查，defer=true。

示例：
用户：「帮我找一些冒险题材的动画」→ {{"mode":"find","defer":false,"name":"","genre":"动画,冒险","year":0,"country":"","sort":"","limit":0}}
用户：「你知道超能陆战队吗」→ {{"mode":"movie_info","defer":false,"name":"超能陆战队","genre":"","year":0,"country":"","sort":"","limit":0}}
用户：「你有没有听说过超能陆战队」→ {{"mode":"movie_info","defer":false,"name":"超能陆战队","genre":"","year":0,"country":"","sort":"","limit":0}}
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


def _extract_unknown_genre_phrase(m: str) -> str | None:
    """抓出用户「想按某类电影找」但不在已知类型表里的那个「类别词」。

    命中条件（全部满足）：
    - 原话是「找电影/推荐」句式（含 推荐/想看/看/找/来/给我/推/介绍 等动词）；
    - 动词后紧跟一个 1~6 字候选词（到 片/电影/题材/类型/风格/系/向/的/标点 为止）；
    - 该候选词既不是标准类型、也不是已知同义词（即 _detect_genres_from_text(cand) 为空）；
    - 且不是泛称/量词/疑问词（电影、好看、什么、一部…）。
    返回该词（供语义检索兜底），否则 None。

    【为何需要】「喜剧」打成「戏剧」、或说了一个库里没有的类别词时，
    _detect_genres_from_text 抽不到任何已知类型 → genre 为空 → find_movies(genre="")
    退化成全局搜索、弱模型凭记忆瞎编。这里把「用户明显在要一个未知类别」识别出来，
    交给语义检索（项目已接 Qwen3-Embedding-8B 向量库）自主纠错/补全接近正确意图的电影。
    """
    _STOP = {"电影", "影片", "片子", "好看的", "好看", "不错", "的", "什么",
             "啥", "几部", "哪些", "一部", "一个", "两个", "这个", "那个", "部", "篇"}
    _PATS = [
        # 动词 + 候选词 + 边界（片/电影/题材/类型/风格/系/向/的）
        r"(?:推荐|想看|要看|看看|看|找|搜|来|给我|整|推|介绍|说说|讲讲|来部|来个|要部|要个|挑|选)\s*([^，。？！、片电影题材类型风格系向的]{1,6}?)\s*(?:片|电影|题材|类型|风格|系|向|的)",
        # 动词 + 候选词 + 句末（无边界词，如「我想看戏剧」）；排除集含 片/电影，避免把泛称「X电影」整段当类别词
        r"(?:推荐|想看|要看|看看|看|找|搜|来|给我|整|推|介绍|说说|讲讲|来部|来个|要部|要个|挑|选)\s*([^，。？！、片电影]{1,8})$",
    ]
    for p in _PATS:
        mt = re.search(p, m)
        if not mt:
            continue
        cand = mt.group(1).strip()
        # 剥掉前置量词（一个/两/部/篇…）
        cand = re.sub(r"^(一|两|二|三|四|五|几|这|那|个|部|篇|些)\s*", "", cand)
        # 剥掉尾部泛称（电影/影片/片子/片），否则「X电影」会被整段当成类别词
        cand = re.sub(r"(电影|影片|片子|片)$", "", cand)
        if not cand or cand in _STOP:
            continue
        # 已知类型 / 同义词 → 不该走语义（那种 genre 非空，根本不会到这步，双保险）
        if _detect_genres_from_text(cand):
            continue
        if 1 <= len(cand) <= 6:
            return cand
    return None


def _extract_params_rule(message: str, sid: str | None = None) -> dict:
    """确定性参数抽取（不依赖 LLM）：覆盖绝大多数「找电影/推荐」句式，速度快、零幻觉、
    不依赖弱模型是否会抽 JSON。作为【电影意图主路径】的参数来源，替代不稳定的 LLM 抽取。

    返回字段与 _extract_params 完全一致，便于 _route_tool 直接消费。

    sid：可选。显式传入会话 id 用于「续轮条件继承」；不传则回退读 _SESSION_ID_CTX。
    必须显式传——流式路径下 FastAPI 把生成器丢进线程池逐 next() 执行，
    contextvar 跨线程丢失，_SESSION_ID_CTX.get() 会退化回 'default'，导致会话记忆（再推荐继承）失效。
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
    # 1.6) 口语点名某部具体电影（库内未知也抽片名，确保仍先走 get_movie_info_by_name 查库）
    #      覆盖「你知道X吗 / X讲什么 / X的导演 / X好看吗 / 介绍一下X」等；
    #      不抽《》（已在 1.5）与库内已知片名（已在 1），只兜底「没书名号且库里没有」的口语，
    #      避免退化成泛化 find_movies 让弱模型凭记忆编造（超能陆战队 bug 根因）。
    uname = _extract_unknown_movie_name(m)
    if uname:
        default["mode"] = "movie_info"
        default["name"] = uname
        return default
    # 1.7) 片名片段搜索：用户记不清完整片名、只记得「标题/片名/名字里有X」「片名带X」等。
    #      必须在 genre 检测（步骤2）【之前】执行，否则「冒险王」里的「冒险」会被当类型，
    #      退化成 find_movies 推评分最高冒险片（用户实测 bug：「标题里有冒险王这三个字」→ 推回安昂传奇）。
    #      命中即走 search_movies 做模糊匹配（如《奇幻变身大冒险》能被「冒险王」命中），而非 genre 检索。
    frag = _extract_title_fragment(m)
    if frag:
        default["mode"] = "name_search"
        default["name"] = frag
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
    if any(k in m for k in ["热度最高", "热度", "最火", "按热度", "受欢迎", "看的人多", "人气"]):
        sorts.append("popularity")
    if any(k in m for k in ["最新", "年份最高", "按年份", "近年", "近期", "新上映"]):
        sorts.append("year")
    if any(k in m for k in ["上映日期", "上映"]):
        sorts.append("release")
    # 随机/随便：明确要「随机/随便/随意推荐」时，用 random 排序（而非默认评分降序，
    # 否则永远返回评分最高的那一部，造成「随机推十次都同一部」的离谱结果）。
    if any(k in m for k in ["随机", "随便", "随意", "随机来", "随便来", "胡乱", "任意挑", "盲选"]):
        sorts.append("random")
    # 去重保序
    default["sort"] = ",".join(dict.fromkeys(sorts))
    # 6) 条数（「篇」是用户常把电影当文章数的口语量词，一并纳入，否则「一篇/两篇」抽不到数→回退5）
    lm = re.search(r"(\d+)\s*[部个篇]", m)
    if lm:
        default["limit"] = int(lm.group(1))
    else:
        # 中文数词（无阿拉伯数字）：「一个 / 一部 / 一篇」→ 1，「两部 / 三部」→ 2 / 3 ……
        # 否则「推荐一个冒险动画」因 limit 抽不到数字会回退成 5，与用户「只要一部」的诉求相悖。
        cm = re.search(r"(十|九|八|七|六|五|四|三|两|二|一)\s*[部个篇]", m)
        if cm:
            _CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
                       "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            default["limit"] = _CN_NUM[cm.group(1)]
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

    # 8) 续轮「再推荐 / 换一个 / 还有别的」：若本轮【未明说】类型/地区/年份/排序，
    #    自动继承上一轮 find 的条件（会话级记忆），解决无状态路由在续轮丢失 genre 的问题
    #    （用户反复吐槽「再推荐脱离前文」）。本轮若显式给了新类型，则以本轮为准，不覆盖。
    #    【语义轮继承】上一轮若是语义检索（mode=semantic），续轮未给任何新条件时继承语义 query，
    #    保持「结局治愈 / 时间循环 / 深夜一个人」这类主题延续，而非退化回评分最高电影。
    if _is_followup(m):
        last = _get_last_find(sid or _SESSION_ID_CTX.get())
        if last:
            if (last.get("mode") == "semantic" and not default["genre"]
                    and not default["sort"] and not default["query"]):
                default["mode"] = "semantic"
                default["query"] = last.get("query") or m
                return default
            if not default["genre"] and last.get("genre"):
                default["genre"] = last["genre"]
            if not default["country"] and last.get("country"):
                default["country"] = last["country"]
            if not default["year"] and last.get("year"):
                default["year"] = last["year"]
            if not default["sort"] and last.get("sort"):
                default["sort"] = last["sort"]
    # 2.5) 语义兜底：用户明显想按「某类电影」找，但抽到的 genre 为空——
    #      说明他要的那个「类」不在已知类型表（错别字/未知分类，如「喜剧」打成「戏剧」、
    #      或说了一个库里没有的类别词）。若仍走 find_movies(genre="") 会退化成全局搜索、
    #      弱模型凭记忆瞎编；正确做法是把原话作为语义描述走 semantic_search_movies
    #      （项目已接 Qwen3-Embedding-8B 向量库），让 RAG 自主纠错/补全接近正确意图的电影。
    #      约束：①genre 确实空（已知/同义词类型已被前序识别，不会到这）；
    #            ②无显式排序意图（带 sort 时尊重排序走 find 全局更合理）；
    #            ③非 defer（「先别急」之类不查库）；④原话里确含一个「未知类别词」。
    if not default["genre"] and not default["sort"] and not default["defer"]:
        unknown = _extract_unknown_genre_phrase(m)
        if unknown:
            default["mode"] = "semantic"
            default["query"] = m  # 用完整原话做语义 query，比孤零零一个词更准
            return default
    return default


# 排序维度 → 中文标签（用于向响应层说明每个分组是按什么排的）
_SORT_LABEL = {
    "rating": "评分最高",
    "popularity": "热度最高",
    "year": "年份",
    "release": "上映日期",
    "random": "随机",
}


def _route_tool(params: dict, message: str, tr=None, sid: str | None = None) -> str:
    """根据结构化参数，在代码里直接调用对应工具，返回工具的真实文本结果。
    message 用于注入 contextvar（供 find_movies 做类型词自动抽取兜底）。
    tr 显式传入用于埋点（记录工具调用链，修复流式异步下记丢的问题）。
    sid 显式传入用于「会话级去重 / 续轮条件继承」（流式线程池下 contextvar 会丢，故显式传）。

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
            result = str(tool.invoke({"query": params.get("query") or message, "top_k": 5}))
            # 语义轮也记入会话记忆（mode=semantic + query），供续轮「再推荐/换一个」继承主题
            _set_last_find(sid or _SESSION_ID_CTX.get(), "", "", 0, "",
                           mode="semantic", query=params.get("query") or message)
            return result

        # 片名片段搜索：用户只记得标题里的字（如「标题里有冒险王这三个字」）→ 走 search_movies 模糊匹配，
        # 而非退化成 genre 检索（「冒险王」含类型词「冒险」会被误判）。
        if mode == "name_search" and params.get("name"):
            tool = _tool_by_name("search_movies")
            if tr is not None:
                tr.tool_calls.append({"name": "search_movies", "args": {"keyword": params["name"]}})
            return str(tool.invoke({"keyword": params["name"]}))

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

        # 会话级「已推荐去重」：本会话之前推过的电影，这次优先排除，
        # 让「再推荐 / 换一个 / 随机」不再命中刚推过的同一部（用户明确吐槽过的点）。
        sid = sid or _SESSION_ID_CTX.get()
        exclude_ids = _recommended_exclude(sid)

        def _run_find_blocks(excl):
            blks = []
            for s in sorts:
                if tr is not None:
                    tr.tool_calls.append({"name": "find_movies", "args": {"genre": genre, "year": year, "country": country, "sort": s, "limit": per_limit, "exclude_ids": list(excl)}})
                raw = str(tool.invoke({
                    "genre": genre,
                    "year": year,
                    "country": country,
                    "sort": s,
                    "limit": per_limit,
                    "exclude_ids": list(excl),
                }))
                label = _SORT_LABEL.get(s, "默认") if s else "评分最高（默认）"
                blks.append(f"【排序：{label}】\n{raw}")
            return blks

        def _append_shortage_note(blks):
            # 诚实告知：仅在「资料库真实不足用户要的数量」时才补一句（如用户要 5 部、库里只有 3 部）。
            # 其余情况（库里充足 / 用户只要 1 部）一律不报总数——报总数对用户体验是噪音，
            # 且极易诱发弱模型把"找到了 M 部"误说成"只有 M 部"（数量幻觉复发点）。
            # （用户明确要求：剩余足够时不用调取数量说明，否则交流体验太差。）
            if requested_limit > 0 and blks:
                m_tot = re.search(r"共 (\d+) 部符合条件", blks[-1])
                true_total = int(m_tot.group(1)) if m_tot else 0
                if 0 < true_total < requested_limit:
                    blks[-1] += (
                        f"\n（注：资料库里符合条件的电影目前共 {true_total} 部，已全部列出，"
                        f"未能凑满你想要的 {requested_limit} 部。）"
                    )

        blocks = _run_find_blocks(exclude_ids)
        _append_shortage_note(blocks)
        combined = "\n\n".join(blocks)

        # 池耗尽兜底：若因排除已推荐而查无结果（该会话把符合条件的电影都推过了），
        # 清空记忆再查一次，保证「还有得推」而非空手而归。
        if exclude_ids and "未找到符合条件" in combined:
            _clear_recommended(sid)
            exclude_ids = set()
            blocks = _run_find_blocks(exclude_ids)
            _append_shortage_note(blocks)
            combined = "\n\n".join(blocks)

        # 记录本次推荐的电影 id，供后续「再推荐 / 随机」继续去重
        _record_recommended(sid, combined)
        # 记录本次 find 的筛选条件，供续轮「再推荐 / 换一个」继承上下文
        # （解决无状态路由在续轮丢失 genre、退化成全局搜索的问题）
        _set_last_find(sid, genre, country, year, params.get("sort") or "")
        return combined
    finally:
        _USER_QUERY_CTX.reset(token)


# 把工具的「真实结果」用自然、会"听人话"的中文讲给用户（事实仍 100% 来自工具，零幻觉）
_RESPOND_TEMPLATE = """你是智影影评网站的智能助手，正在和用户多轮聊电影。

【直接进入正题 · 不要寒暄】
- 用户直接问电影/要推荐时，**开头不要打招呼、不要自我介绍、不要寒暄**（如「你好呀～我是智影影评网站的智能助手…」这类）。
- 即使上文（对话历史）里有过问候，也不要重复——直接回答当前问题即可。

【铁律 · 只能基于工具结果说话】
- 你只能依据下面【工具结果】里给出的信息来谈论电影。禁止编造任何电影名、评分、年份、类型、地区、简介。
- 若【工具结果】是「未找到 / 没有符合条件的电影」，就如实、友好地告诉用户资料库里暂时没有这部电影（可能还没收录），并可以友好地问一句「你想看什么类型的？我帮你推荐类似的」。
- 若【工具结果】是一组电影，就用自然、亲切的口吻介绍这批电影（可加一句引导语，例如「给你挑了几部符合要求的电影：」），不要生硬复述原始格式；保留片名、评分、类型等关键信息即可。
- **禁止自行编造数量声明（弱模型极易在此幻觉，务必遵守）**：你**绝不可以**自己冒出「资料库里这类电影目前只有 X 部，都列给你了」之类的句子；**尤其当你本次只列出了 M 部、但【工具结果】里的系统提示说共有 N 部（N>M）时，绝不可说「只有 M 部」**——那会把"找到了 M 部"误说成"总共只有 M 部"，是典型数量幻觉。只有当【工具结果】里出现了系统生成的提示时，你才按提示里的数字如实转达：①「（注：…共 M 部…未能凑满你想要的 N 部）」**短缺提示**→自然说「资料库里这类电影目前只有 M 部，都列给你了」（M 以提示数字为准）；②「（注：…共 N 部，已为你列出其中的 M 部）」**充足提示**→自然说「资料库里这类电影共 N 部，我先给你列了评分最高的 M 部」（N、M 均以提示数字为准，**充足时绝不可省略成「只有 M 部」**）。用户要的数量已满足且【工具结果】里无任何上述提示时，直接自然介绍电影即可，**不要画蛇添足补数量声明**。绝不要编造资料库里没有的片名/评分/年份；补充资料库外电影须标注「（资料库外 · 仅供参考）」。优先做法：先如实交差资料库里的电影。**再次强调**：【工具结果】里已经写明了符合条件的真实总数（形如「资料库里符合条件的电影目前共 N 部」），你直接使用这个数字即可；**绝对禁止自己另写「共 X 部」「只有 X 部」「都列给你了」之类的总结数量句**——那会被判定为编造幻觉。
- **两条硬性输出要求（违反即视为严重失误）**：(a) **用户只要「一部」电影时（如「推荐一部/来一部/随便来一部」），【工具结果】里不会出现任何数量提示，你也绝对不要自行补充「资料库里共 N 部」「都列给你了」之类的总结**——直接自然介绍那一部电影即可，报总数对用户毫无意义且极易误导（例：只推了一部却说"共7部都列给你了"是完全错误、毫无意义的）。(b) **无论何种情况，【工具结果】里给出的每一部电影，你都必须把片名（至少）说出来**，绝不可只复述末尾的数量提示而把电影本身省略掉。尤其是用户说「再推荐一个/换一个/再来一部」时，必须把新推出来的那部电影名字点出来，不能只回一句数量、让对话看起来"没有推荐任何电影"。
- 若【工具结果】是一段「单部电影资料」（以《片名》开头、含导演/演员/评分/简介等字段），你必须介绍【那部电影】本身。特别注意：即使用户的话里出现了「你的名字」这类措辞，只要工具结果是一份电影资料，就说明用户在问那部【电影】（例如《你的名字。》），你只能介绍该电影；**绝不要**把它理解成在问「助手你叫什么名字」，也**绝不要**输出任何关于「助手身份 / 助手名字 / 我是谁」的内容。
- 若【工具结果】包含多个以「【排序：xxx】」开头的分组，说明是按不同维度（如评分最高 / 热度最高）分别查到的电影。请【分别、逐一】介绍每个分组，并明确点出该组是按什么维度排的（如「这是评分最高的一部」「这是热度最高的一部」）；不同分组往往是不同的电影，绝对不要把它们混为一谈、也不要只介绍其中一个而漏掉其它分组。
- 【工具结果】里的「【排序：xxx】」只是内部分组标记，组织答案时不要原样照抄这些标记，用自然说法带出（如「按评分排，最高的是…」「按热度排，最高的是…」）。若某组末尾出现了系统生成的提示——「（注：资料库里…共 N 部…未能凑满你想要的 M 部）」**短缺提示** 或「（注：资料库里…共 N 部，已为你列出其中的 M 部）」**充足提示**——必须**严格按提示里的 N 数字**自然地转达给用户（短缺→「资料库里这类电影目前只有 N 部，都列给你了」；充足→「资料库里这类电影共 N 部，我先给你列了 M 部」，**充足时绝不可说「只有 M 部」**）；数字一律以提示为准，不得自行改动；【工具结果】里没有此类提示时，不要自己添加任何数量声明。
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


# —— 纯闲聊模板 ——
# 用户来聊天/倾诉（不是找电影）时使用。此前直接用 movie 导向的 system_prompt() 给弱模型，
# 它在 temp=0 下要么照搬「不客气～有需要随时找我」这类回应感谢的收尾、要么干脆罗列一屏电影，
# 都不懂「陪我聊两句」= 聊天。故用专门的闲聊模板明确指令，让模型真正「接住话」。
_CHAT_TEMPLATE = """你是智影影评网站的智能助手，现在用户是在【闲聊/倾诉/日常对话】，不是在找电影。
系统已确认：用户刚才的话里没有任何电影、影评或推荐需求（有的话会自动走电影检索，不会走到你这里）。

请用自然、有温度的中文回应：
- 认真接住对方的话：对方说累/心情不好/想聊天，就先共情、顺着聊（关心一句、聊聊怎么放松都行），不要答非所问。
- 结尾可以自然反问一句，把对话延续下去。
- 【绝对禁止】推荐、罗列或介绍任何电影，不要提「资料库/数据库/检索」，不要聊影评。
- 【绝对禁止】用「不客气」「有需要随时找我」「拜拜」「感谢你」这类【回应感谢/道别】的收尾语——
  那是用户说「谢谢」时才用的；用户现在是想聊天，你要聊下去，而不是结束对话。
- 保持简短自然（1~3 句）。

【对话上文】
{history}

【用户刚说的话】
{question}
"""


def _respond_chat(message: str, history) -> str:
    prompt = (_CHAT_TEMPLATE.replace("{history}", _fmt_history(history))
                            .replace("{question}", message))
    return _llm_reply(prompt, [], message)


def _respond_chat_stream(message: str, history):
    """_respond_chat 的流式版：逐 token 产出闲聊回复。"""
    prompt = (_CHAT_TEMPLATE.replace("{history}", _fmt_history(history))
                            .replace("{question}", message))
    yield from _llm_reply_stream(prompt, [], message)


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


# ---------------------------------------------------------------------------
# 数量声明清洗：弱模型偶尔不听 prompt 铁律，自编「资料库共 N 部 / 都列给你了」。
# 仅当数字与工具真实总数不符时才整句删除，避免误伤代码追加的诚实短缺提示。
# 诚实提示格式为「资料库里符合条件的电影目前共 M 部…」（不以「这类/目前共」开头），
# 不会被下面的正则命中，故安全保留。
# ---------------------------------------------------------------------------
_COUNT_CLAIM_RE = re.compile(
    r"资料库里(?:这[类种]|目前)?(?:的)?(?:电影|影片)?(?:目前)?共\s*\d+\s*部"
    r"(?:，?都列给?你(?:了|出来)?|，?已全部?列出|，?都列出来)?。?"
)


def _true_total_from_raw(raw: str) -> int:
    """从工具结果抽真实总条数（find_movies 返回『共 X 部符合条件』）。"""
    m = re.search(r"共\s*(\d+)\s*部符合条件", raw or "")
    return int(m.group(1)) if m else 0


def _sanitize_polish(answer: str, true_total: int) -> str:
    """剔除润色 LLM 自编的数量声明；数字与工具真实总数一致（诚实提示）则保留。"""
    if true_total <= 0:
        return answer

    def _repl(mm):
        num = re.search(r"共\s*(\d+)\s*部", mm.group(0))
        if num and int(num.group(1)) != true_total:
            return ""  # 数字对不上 → 判定为编造，整句删除
        return mm.group(0)  # 数字吻合（代码诚实提示）→ 保留

    cleaned = _COUNT_CLAIM_RE.sub(_repl, answer)
    # 删除后可能留下句首游离标点（如「。给你挑了几部…」），清理掉
    return re.sub(r"^[。，、\s]+", "", cleaned).strip()


# 纯问候/自介的助手消息（如 _GREET_REPLY）在「电影答案润色」时不传给模型——
# 否则润色模型看到上一条是问候，会把问候原样重复在新回答开头，
# 导致「用户直接问电影」却先被寒暄一遍（用户实测：先「你好」再问推荐，开头又冒出
# 「你好呀～我是智影影评网站的智能助手…」）。问候是纯客套、无电影信息，丢掉不影响上下文。
_GREET_ONLY_RE = re.compile(r"^(?:你好|您好|哈喽|嗨)[^，。]{0,8}?我是.*(?:智能助手|电影助手).*$")


def _strip_greeting_only(history):
    """从对话历史里去掉「纯问候/自介」的助手消息（仅用于电影答案润色环节）。"""
    if not history:
        return history
    return [m for m in history
            if not (isinstance(m, AIMessage) and _GREET_ONLY_RE.match(str(m.content or "")))]


def _respond_result(message: str, raw_result: str, history, true_total: int = 0) -> str:
    prompt = (_RESPOND_TEMPLATE.replace("{history}", _fmt_history(_strip_greeting_only(history)))
                                .replace("{question}", message)
                                .replace("{result}", raw_result))
    answer = _llm_reply(prompt, [], message)
    return _sanitize_polish(answer, true_total)


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
        # 注意：本函数只做「工具结果的格式化」，不干预模型自主选/调工具的过程（function calling 由上面循环负责）。
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
        # 最终答案用确定性模板把【工具真实结果】自然复述（不带工具绑定 → 不会泄工具语法、零幻觉）
        # 注意：本函数只做「工具结果的格式化」，不干预模型自主选/调工具的过程（function calling 由上面循环负责）。
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


# —— 库外具体电影：方案1 回复（公开知识作答 + 代码强制免责声明）——
# 设计前提：get_movie_info_by_name 已经先被调用且返回「未找到」，即已确认资料库无该片（DB-first 不变）。
# 之后允许模型基于公开知识介绍，但【免责声明由代码追加】，不依赖模型自觉，确保一定出现、不会被漏掉。
_NOT_FOUND_KNOWLEDGE_DISCLAIMER = "（注：资料库暂未收录《{name}》，以下内容由模型基于公开知识生成，仅供参考。）"

_NOT_FOUND_KNOWLEDGE_TEMPLATE = """你是一个电影知识助手。用户问的是《{name}》。

资料库里目前没有收录这部电影，你无法从资料库获取它的官方数据。请基于你的公开知识，用中文简要、自然地介绍这部电影，帮助用户：
- 一句话定位（如「《{name}》是 Y 年 Z 国的一部 A 类型电影」）
- 导演、主要演员
- 剧情梗概（2-3 句，不剧透关键反转）
- 口碑/评价（如获奖、影评共识；可以说「口碑不错」，但不要编造具体评分/票房数字，除非你非常确定）
- 风格与适合人群（1 句）

注意（务必遵守）：
1. 不要说「资料库里有这部电影」「根据资料库」之类的话——它根本不在资料库里。
2. 不要编造你不确定的人物姓名、具体票房/评分数字；不确定就笼统表述。
3. 只介绍《{name}》这一部，不要跑题去推荐别的电影（除非用户明确要求）。
4. 语气像朋友安利，自然友好，不要列成生硬的条目堆。

用户原话：{question}
对话历史：
{history}
"""


def _not_found_knowledge_reply(name: str, message: str, history) -> str:
    """库外具体电影（非流式）：模型用公开知识作答，代码强制追加免责声明，返回完整字符串。"""
    prompt = (_NOT_FOUND_KNOWLEDGE_TEMPLATE.replace("{name}", name or "")
              .replace("{history}", _fmt_history(_strip_greeting_only(history)))
              .replace("{question}", message))
    ans = "".join(_llm_reply_stream(prompt, [], message))
    return ans + "\n\n" + _NOT_FOUND_KNOWLEDGE_DISCLAIMER.format(name=name or "")


def _not_found_knowledge_reply_stream(name: str, message: str, history):
    """库外具体电影（流式）：逐 token 产出模型作答，结束时代码强制追加免责声明。"""
    prompt = (_NOT_FOUND_KNOWLEDGE_TEMPLATE.replace("{name}", name or "")
              .replace("{history}", _fmt_history(_strip_greeting_only(history)))
              .replace("{question}", message))
    for t in _llm_reply_stream(prompt, [], message):
        yield t
    yield "\n\n" + _NOT_FOUND_KNOWLEDGE_DISCLAIMER.format(name=name or "")


def _deterministic_movie_path(message: str, history, tr=None, sid: str | None = None) -> str:
    """电影意图【主路径】：用我们已验证可靠的「代码抽参 + 代码调工具 + 自然语言」路径（仍零幻觉）。
    代码直接调工具（不再依赖弱模型会不会 function calling），100% 可靠、只多 1 次 LLM 润色。
    sid 显式传入会话 id（流式线程池下 contextvar 会丢，必须显式传才能保住会话记忆）。
    """
    forced_title = _find_title_in_text(message)
    if forced_title:
        # 从用户原话确定性认出已知片名 → 强制按该电影查库，绕开弱模型把「你的名字」当问助手名的偏见
        tool = _tool_by_name("get_movie_info_by_name")
        if tr is not None:
            tr.tool_calls.append({"name": "get_movie_info_by_name", "args": {"name": forced_title}})
        raw = str(tool.invoke({"name": forced_title}))
    else:
        params = _extract_params_rule(message, sid)
        if params.get("defer"):
            return _respond_defer(message, history)
        raw = _route_tool(params, message, tr, sid)
    if _is_not_found(raw):
        # 库外具体电影（用户点名、资料库未收录）→ 方案1：公开知识作答 + 代码强制免责声明
        nm = forced_title or (params.get("name") if (params and params.get("mode") == "movie_info") else None)
        if nm:
            return _not_found_knowledge_reply(nm, message, history)
        return _not_found_reply(forced_title)
    return _respond_result(message, raw, history, _true_total_from_raw(raw))


def _run_deterministic_stream(message: str, history, tr=None, sid: str | None = None):
    """_deterministic_movie_path 的流式版：确定性工具调用（同步）照常执行，
    最终自然语言用 stream=True 逐 token 产出。是电影意图的【默认流式路径】。

    流程：规则抽参（0 次 LLM）→ 代码调工具（真实 DB 数据）→ 模板复述（1 次 LLM 润色）。
    相比自主 agent 循环（最坏 3 轮 × LLM + 兜底 2 次 LLM = 5 次调用），
    这里固定 1 次 LLM 调用，延迟从 ~50s 降到 ~10s，且答案 100% 基于工具真实数据。
    sid 显式传入会话 id（流式线程池下 contextvar 会丢，必须显式传才能保住会话记忆）。
    """
    forced_title = _find_title_in_text(message)
    if forced_title:
        tool = _tool_by_name("get_movie_info_by_name")
        if tr is not None:
            tr.tool_calls.append({"name": "get_movie_info_by_name", "args": {"name": forced_title}})
        raw = str(tool.invoke({"name": forced_title}))
        if _is_not_found(raw):
            # 库外具体电影（已知片名但查库落空，罕见）→ 方案1：公开知识作答 + 代码强制免责声明
            yield from _not_found_knowledge_reply_stream(forced_title, message, history)
            return
        yield from _respond_result_stream(message, raw, history)
        return
    params = _extract_params_rule(message, sid)
    if params.get("defer"):
        # 用户说"稍后再给要求 / 先别急"——不查库，自然回应等他开口
        prompt = (_DEFER_TEMPLATE.replace("{history}", _fmt_history(history))
                                      .replace("{question}", message))
        for t in _llm_reply_stream(prompt, [], message):
            yield t
        return
    raw = _route_tool(params, message, tr, sid)
    if _is_not_found(raw):
        # 库外具体电影（用户点名、资料库未收录）→ 方案1：模型用公开知识作答 + 代码强制免责声明
        nm = params.get("name") if params.get("mode") == "movie_info" else None
        if nm:
            yield from _not_found_knowledge_reply_stream(nm, message, history)
        else:
            yield _not_found_reply(None)
        return
    yield from _respond_result_stream(message, raw, history)


def _respond_result_stream(message: str, raw_result: str, history):
    """_respond_result 的流式版：真·逐 token 产出（恢复实时流式）。

    关键教训：上一版为做「数量清洗」把 LLM 整段输出先 ``"".join`` 缓冲、清洗完再按 2 字/块吐，
    导致电影/推荐这条最常用路径**完全不实时流式**（长空白 → 再 2 字蹦），用户感知为「流式失效」。
    现改为直接 ``yield from _llm_reply_stream``：LLM 每产出一个 token 就立刻推给前端。
    数量幻觉改「源头预防」：_RESPOND_TEMPLATE 已禁止模型自编「共 N 部」，且工具结果里
    已含真实总数（第 588 行注入「共 M 部符合条件」），模型直接照说即可，无需事后缓冲清洗。
    非流式 _respond_result 仍保留 _sanitize_polish 硬兜底。
    """
    prompt = (_RESPOND_TEMPLATE.replace("{history}", _fmt_history(_strip_greeting_only(history)))
                                .replace("{question}", message)
                                .replace("{result}", raw_result))
    yield from _llm_reply_stream(prompt, [], message)


def _deterministic_movie_path_stream(message: str, history, tr=None, sid: str | None = None):
    """_deterministic_movie_path 的流式版：确定性工具调用（同步）照常执行，
    但最终自然语言用 stream=True 逐 token 产出，避免兜底时「一次性全吐」。

    作为「主路径（_run_deterministic_stream）也空手而归」时的最后兜底（罕见，如工具全失败）。
    sid 显式传入会话 id（流式线程池下 contextvar 会丢，必须显式传才能保住会话记忆）。
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
        raw = _route_tool(params, message, tr, sid)
    if _is_not_found(raw):
        # 库外具体电影（用户点名、资料库未收录）→ 方案1：公开知识作答 + 代码强制免责声明
        nm = forced_title or (params.get("name") if (params and params.get("mode") == "movie_info") else None)
        if nm:
            yield from _not_found_knowledge_reply_stream(nm, message, history)
            return
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
    sid_token = _SESSION_ID_CTX.set(session_id)  # 供「会话级已推荐去重」识别会话
    try:
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
                answer = _deterministic_movie_path(message, history, tr, session_id)
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
        return answer, meta
    finally:
        _safe_reset(uq_token)
        _safe_reset(trace_token)
        _safe_reset(sid_token)


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
    sid_token = _SESSION_ID_CTX.set(session_id)  # 供「会话级已推荐去重」识别会话
    try:
        intent = _classify_intent(message)
        store = _get_history(session_id)
        history = list(store.messages)
        full = ""

        if intent == "chat":
            # 纯问候确定性返回（不调 LLM，~0ms）；其余闲聊照常流式 LLM（用专门的闲聊模板）
            if _is_pure_greeting(message):
                full = _GREET_REPLY
                yield {"type": "token", "text": full}
            else:
                for t in _respond_chat_stream(message, history):
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
                for t in _run_deterministic_stream(message, history, tr, session_id):
                    full += t
                    yield {"type": "token", "text": t}
                if not full:
                    # 确定性路径空手而归（罕见，如工具全失败）→ 退回自主 agent 作最后尝试
                    for t in _agent_run_with_guardrail_stream(message, history, tr):
                        full += t
                        yield {"type": "token", "text": t}
                    if not full:
                        # 仍为空 → 再退回确定性兜底
                        for t in _deterministic_movie_path_stream(message, history, tr, session_id):
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
        yield {"type": "done", "meta": meta}
    finally:
        _safe_reset(uq_token)
        _safe_reset(trace_token)
        _safe_reset(sid_token)
