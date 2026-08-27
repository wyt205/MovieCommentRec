# -*- coding: utf-8 -*-
"""
电影推荐 Agent 评测运行器
================================
目的：把"抗幻觉"从嘴上说法变成可量化数据。
用法：python run_eval.py            （进程内直连，需能访问 LLM_API_KEY）
      python run_eval.py --quiet    （只打分数卡，不打每条回答）
      python run_eval.py --temperature 0.3   （临时用非确定性温度跑；默认 0 基线）

评分信号（均为可解释、可辩护的启发式，非黑箱）：
  - 可溯源(library)：问库内电影，回答必须出现该电影标题（=确实调了工具拿到真实数据）。
  - 库外诚实(refusal)：问库外电影，回答必须包含「资料库未收录」类明确声明（代码强制追加），
                    且不得把公开知识谎称为「根据资料库/资料库里有」。
                    【2026-08 更新】库外电影已不再是「拒答」：先查库确认无 → 模型基于公开知识
                    简要介绍 + 代码强制追加免责声明（「资料库暂未收录《X》，以下内容由模型基于
                    公开知识生成，仅供参考」）。因此旧评分「出现导演/剧情=越界泄漏=失败」已废弃，
                    改为「必须有免责声明 且 不得谎称数据来自资料库」。
  - 幻觉率 = 库外样本里「谎称数据来自资料库/编造资料库有该片」的条数 / 库外样本总数。
  - 类型抽取(genre)：返回结果里必须出现预期标准类型（刺激→动作 等）的电影。
  - 多维度(multi_dim)：评分最高 与 热度最高 必须返回两部不同电影。
  - 多轮记忆(memory_followup)：同一会话连续两轮，续轮「再推荐/换一个」未明说条件时，
                   必须继承上一轮类型（不退化回评分最高电影），且推的是不同的一部。
  - 标题片段(title_fragment)：问「标题里带X字」，回答里须出现含该片段的库内片名
                      （=路由到按片名搜索，验证更聪明的片段路由能力）。

注意：库内/库外用 DB 实时判定，换片或扩库后无需改用例。
语义类(semantic)为观察项，不计入幻觉率分母（避免误判）。
评测严谨化：默认固定 temperature=0 跑基线（关闭随机性 → 输出确定性、可复现，
分数稳定、两次跑可直接对比；免费 GLM-4-Flash 非确定性，不固定温度分数会抖动）。
可用 --temperature 临时覆盖。
"""
import json
import os
import re
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 评测严谨化：固定 temperature 基线（确定性、可复现）──
# 默认 0（关闭随机性，分数稳定可对比）；可用 `--temperature 0.3` 临时覆盖。
# 必须在本文件 import app.ai.agent 之前设好环境变量，否则 _make_llm() 已按默认 0.3 固化。
_eval_args = sys.argv[1:]
QUIET = "--quiet" in _eval_args
_eval_temperature = "0"
for _i, _a in enumerate(_eval_args):
    if _a == "--temperature" and _i + 1 < len(_eval_args):
        _eval_temperature = _eval_args[_i + 1]
os.environ["AGENT_LLM_TEMPERATURE"] = _eval_temperature

from sqlalchemy import select
from app.db.database import SessionLocal
from app.models import Movie
from app.ai.agent import chat

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "eval_cases.json")
RESULT_PATH = os.path.join(HERE, "eval_result.json")

# 每次评测运行使用独立的会话后缀，避免上一次运行遗留的会话历史（chat_sessions 表）
# 污染本轮：否则同一 eval-<id> 会话跨轮累积历史，会让同一条用例在两次运行间表现不一致。
_RUN_TOKEN = f"{os.getpid()}-{int(time.time() * 1000) % 1000000}"

# 库外电影【新行为】：模型基于公开知识介绍 + 代码强制追加免责声明（不是拒答）。
# 评分改为「必须有明确免责声明 / 不得谎称数据来自资料库」。
# 免责声明标记：答案里出现 未收录/未找到/资料库暂时没有 等任一即视为「已声明」。
DISCLAIMER_RE = re.compile(
    r"资料库暂未收录|资料库(?:里|中|内)?暂时?没有|未收录|没收录|未找到|"
    r"查不到|暂无.{0,4}(?:收录|找到)|没有.{0,8}(?:记录|信息|收录|这部电影|找到)"
)
# 谎称数据来自资料库：把公开知识包装成「根据资料库/资料库里有」→ 视为虚假声明
FALSE_DB_CLAIM_RE = re.compile(
    r"根据资料库|依据资料库|资料库(?:里|中|内)?(?:确实|绝对|肯定|有|收录了|查到了)|"
    r"这是?资料库(?:里|中|内)?(?:的|收录)?"
)
# 闲聊不应用「回应感谢/道别」的收尾语（用户是来聊天/倾诉的，不是来道谢的）。
# 只查「没提电影」太宽松——像「不客气～有需要随时找我🙂」也能蒙混过关，实际是答非所问。
_CHAT_CLOSING_RE = re.compile(
    r"不客气|有需要随时找我|随时找我|有事再找我|拜拜|再见|下次再聊|"
    r"(?:谢谢|感谢)你的(?:分享|反馈|使用|提问)"
)

def load_movies():
    db = SessionLocal()
    rows = db.execute(select(Movie)).scalars().all()
    db.close()
    out = []
    for m in rows:
        out.append({
            "id": m.id,
            "title": m.title,
            "year": m.year,
            "rating": m.rating,
            "popularity": m.popularity,
            "genres": [g.strip() for g in (m.genres or "").split(",") if g.strip()],
        })
    return out


MOVIES = load_movies()


def match_movie(target):
    """库内是否存在与该 target 匹配的电影（双向子串，兼容《你的名字》vs《你的名字。》）。"""
    for m in MOVIES:
        if target in m["title"] or m["title"] in target:
            return m
    return None


def titles_in_answer(answer):
    """回答里出现的、且确实在库内的电影标题。"""
    return [m["title"] for m in MOVIES if m["title"] in answer]


def genre_titles(genre):
    return {m["title"] for m in MOVIES if genre in m["genres"]}


def is_list_answer(answer):
    """回答是否被组织成了「推荐列表」（≥2 部不同的库内电影）。
    用于具体电影问答（library/fuzzy/title_fragment）：问「某部电影讲什么/叫什么」时，
    正确答案应【直接围绕那一部】展开；出现 ≥2 部不同电影说明退化成批量推荐，判失败，
    防止「答案里碰巧出现目标片名」的假通过（用户实测 F01 即此：问奇幻大冒险却列了 3 部）。"""
    return len(titles_in_answer(answer)) >= 2


def score(case, answer, prev_answer=None):
    t = case["type"]
    res = {"id": case["id"], "type": t,
           "query": case.get("query") or " -> ".join(case.get("turns", [])),
           "pass": None, "note": "", "answer": answer,
           "_hallucinated": False, "_refused": None}

    if t == "library":
        m = match_movie(case["target"])
        if not m:
            res["pass"] = None
            res["note"] = "(库内判定失败：target 未匹配到库内电影，已跳过)"
            return res
        is_list = is_list_answer(answer)
        hit = m["title"] in answer
        res["pass"] = hit and not is_list
        res["note"] = (f"期望直接答《{m['title']}》：{'命中' if hit else '未命中'} / "
                       f"退化成列表={'是' if is_list else '否'}")

    elif t == "refusal":
        m = match_movie(case["target"])
        if m:  # 该 target 实际在库内 → 自动重判为库内样例
            is_list = is_list_answer(answer)
            hit = m["title"] in answer
            res["type"] = "library(auto)"
            res["pass"] = hit and not is_list
            res["note"] = f"运行时发现《{m['title']}》在库内，已按库内判定：{'命中' if hit else '未命中'} / 退化成列表={'是' if is_list else '否'}"
            return res
        # 库外电影：新行为 = 公开知识作答 + 明确免责声明（代码强制追加）
        disclaimed = bool(DISCLAIMER_RE.search(answer))
        false_claim = bool(FALSE_DB_CLAIM_RE.search(answer))
        mentions_target = case["target"] in answer
        res["pass"] = disclaimed and not false_claim
        res["note"] = (f"免责声明={'是' if disclaimed else '否'} / "
                       f"谎称库内={'是' if false_claim else '否'} / "
                       f"提及目标片名={'是' if mentions_target else '否'}")
        res["_hallucinated"] = false_claim
        res["_refused"] = disclaimed

    elif t == "chat":
        mentioned = titles_in_answer(answer)
        closing = bool(_CHAT_CLOSING_RE.search(answer))
        # 去掉空白/标点后剩余字数过少 = 没接住话（如只有「好的」「嗯」）
        short = len(re.sub(r"[\s，。！？、~～\-–—…]+", "", answer or "")) < 4
        res["pass"] = (len(mentioned) == 0) and not closing and not short
        res["note"] = (f"闲聊应自然接话：提到电影={mentioned or '无'} / "
                       f"收尾式(把聊天当道谢)={'是' if closing else '否'} / 内容过短={'是' if short else '否'}")

    elif t == "identity":
        ok = ("llm" in answer.lower()) or ("助手" in answer) or ("我是" in answer)
        res["pass"] = ok
        res["note"] = f"身份自介标记={'命中' if ok else '未命中'}"

    elif t == "defer":
        mentioned = titles_in_answer(answer)
        res["pass"] = (len(mentioned) == 0)
        res["note"] = f"暂缓不应列电影，实际提到：{mentioned or '无'}"

    elif t == "genre":
        expected = genre_titles(case["expect_genre"])
        hit = set(titles_in_answer(answer)) & expected
        res["pass"] = len(hit) > 0
        res["note"] = f"期望类型「{case['expect_genre']}」，命中：{list(hit) or '无'}"

    elif t == "multi_dim":
        rating_top = max(MOVIES, key=lambda x: x["rating"] or 0)
        pop_top = max(MOVIES, key=lambda x: x["popularity"] or 0)
        # 用「唯一数值 OR 标准片名」判断，规避模型转写片名/省略数值导致的误判
        rating_present = (str(rating_top["rating"]) in answer) or (rating_top["title"] in answer)
        pop_present = (str(pop_top["popularity"]) in answer) or (pop_top["title"] in answer)
        ok = rating_present and pop_present and (rating_top["title"] != pop_top["title"])
        res["pass"] = ok
        res["note"] = (f"评分最高《{rating_top['title']}》(评分{rating_top['rating']}) / "
                       f"热度最高《{pop_top['title']}》(热度{pop_top['popularity']}) "
                       f"是否都出现：{ok}")

    elif t == "memory_followup":
        # 多轮记忆：同一会话连续两轮，续轮「再推荐/换一个」必须继承上一轮类型，
        # 且不能推回与首轮同一部（去重生效）。prev_answer 为第一轮回答。
        expected = genre_titles(case["expect_genre"])
        hit = set(titles_in_answer(answer)) & expected
        same = False
        if prev_answer:
            same = bool(set(titles_in_answer(prev_answer)) & set(titles_in_answer(answer)))
        res["pass"] = (len(hit) > 0 and not same)
        res["note"] = (f"续轮应保持「{case['expect_genre']}」，命中：{list(hit) or '无'}；"
                       f"与首轮{'重复' if same else '为不同电影'}，去重{'失败' if same else '生效'}")

    elif t == "fuzzy":
        m = match_movie(case["target"])
        hit = bool(m) and ((case["target"] in answer) or (m["title"] in answer))
        is_list = is_list_answer(answer)
        res["pass"] = hit and not is_list
        res["note"] = (f"模糊匹配→《{m['title'] if m else case['target']}》：{'命中' if hit else '未命中'} / "
                       f"退化成列表={'是' if is_list else '否'}")

    elif t == "title_fragment":
        frag = case["fragment"]
        m = match_movie(frag)
        hit = bool(m) and ((frag in answer) or (m["title"] in answer))
        is_list = is_list_answer(answer)
        res["pass"] = hit and not is_list
        res["note"] = (f"标题片段「{frag}」→《{m['title'] if m else '?'}》：{'命中' if hit else '未命中'} / "
                       f"退化成列表={'是' if is_list else '否'}")

    elif t == "semantic":
        res["pass"] = None
        res["note"] = "观察项（需人工看是否真·语义相关/是否诚实拒答）"

    return res


def main():
    quiet = "--quiet" in sys.argv
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    refusal_total = refusal_ok = hallucination = leakage = 0
    cat_stats = {}

    for case in cases:
        sid = f"eval-{case['id']}-{_RUN_TOKEN}"
        # 多轮记忆类：同一会话按 turns 顺序连发（真实测「会话记忆/续轮继承」），
        # 用最后一轮答案评分；其余用例单轮即可。
        if case.get("turns"):
            answers = [chat(turn, sid) for turn in case["turns"]]
            answer = answers[-1]
            r = score(case, answer, prev_answer=answers[0] if len(answers) > 1 else None)
        else:
            answer = chat(case["query"], sid)
            r = score(case, answer)
        results.append(r)

        if not quiet:
            tag = "✅通过" if r["pass"] else ("⚠️观察" if r["pass"] is None else "❌未过")
            print(f"[{case['id']}] {r['type']:>12} | {r['query']}")
            print(f"      -> {answer[:110]}")
            print(f"      {tag}  {r['note']}\n")

        if r["_refused"] is not None:  # 真正的库外拒答
            refusal_total += 1
            if r["_refused"] and not r["_hallucinated"]:
                refusal_ok += 1
            if r["_hallucinated"]:
                hallucination += 1
                leakage += 1
        if r["pass"] is not None:
            c = r["type"]
            s = cat_stats.setdefault(c, [0, 0])
            s[0] += 1
            if r["pass"]:
                s[1] += 1

        time.sleep(1.3)  # 避免免费档限流

    total_scored = sum(s[0] for s in cat_stats.values())
    total_pass = sum(s[1] for s in cat_stats.values())
    hallu_rate = (hallucination / refusal_total) if refusal_total else 0.0
    refuse_acc = (refusal_ok / refusal_total) if refusal_total else 0.0

    summary = {
        "temperature_基线": _eval_temperature,
        "样本总数": len(results),
        "有效评分样本": total_scored,
        "总通过率": f"{(total_pass / total_scored * 100):.1f}%" if total_scored else "N/A",
        "库外样本数": refusal_total,
        "库外诚实声明率": f"{refuse_acc * 100:.1f}%",
        "幻觉率_库外谎称来自资料库": f"{hallu_rate * 100:.1f}%",
        "越界泄漏率_库外却补外部知识": f"{(leakage / refusal_total * 100):.1f}%" if refusal_total else "N/A",
        "分类通过率": {c: f"{s[1]}/{s[0]}" for c, s in cat_stats.items()},
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("评测分数卡")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print(f"\n详细结果已写入：{RESULT_PATH}")


if __name__ == "__main__":
    main()
