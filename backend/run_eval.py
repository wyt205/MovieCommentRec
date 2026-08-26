# -*- coding: utf-8 -*-
"""
电影推荐 Agent 评测运行器
================================
目的：把"抗幻觉"从嘴上说法变成可量化数据。
用法：python run_eval.py            （进程内直连，需能访问 LLM_API_KEY）
      python run_eval.py --quiet    （只打分数卡，不打每条回答）

评分信号（均为可解释、可辩护的启发式，非黑箱）：
  - 可溯源(library)：问库内电影，回答必须出现该电影标题（=确实调了工具拿到真实数据）。
  - 拒答(refusal)：问库外电影，回答应含"未找到/没收录"类表述，且不得为该片凭空编造
                  导演/主演/年份/评分等具体事实。
  - 幻觉率 = 库外样本里"编造了具体事实"的条数 / 库外样本总数。
  - 类型抽取(genre)：返回结果里必须出现预期标准类型（刺激→动作 等）的电影。
  - 多维度(multi_dim)：评分最高 与 热度最高 必须返回两部不同电影。
  - 否定类型(neg_genre)：返回结果里不得出现被排除的类型（不要动画）。

注意：库内/库外用 DB 实时判定，换片或扩库后无需改用例。
语义类(semantic)为观察项，不计入幻觉率分母（避免误判）。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from app.db.database import SessionLocal
from app.models import Movie
from app.ai.agent import chat

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "eval_cases.json")
RESULT_PATH = os.path.join(HERE, "eval_result.json")

REFUSAL_RE = re.compile(r"没有.{0,8}(记录|信息|收录|这部电影|找到)|未找到|未收录|没收录|查不到|暂无.{0,4}(收录|找到)|资料库(里|中)暂时没有")
# 越界知识泄漏：库外电影已拒答，却仍用记忆补了导演/剧情等具体事实
ATTR_RE = re.compile(r"(导演|执导|主演|上映于|上映于\s*\d{4}|年由|评分高达|评分\d|豆瓣\d|讲述了|故事讲述|是一部.{0,12}电影|剧情简介)")


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


def score(case, answer):
    t = case["type"]
    res = {"id": case["id"], "type": t, "query": case["query"], "pass": None,
           "note": "", "answer": answer, "_hallucinated": False, "_refused": None}

    if t == "library":
        m = match_movie(case["target"])
        if not m:
            res["pass"] = None
            res["note"] = "(库内判定失败：target 未匹配到库内电影，已跳过)"
            return res
        ok = m["title"] in answer
        res["pass"] = ok
        res["note"] = f"期望出现《{m['title']}》，实际{'命中' if ok else '未命中'}"

    elif t == "refusal":
        m = match_movie(case["target"])
        if m:  # 该 target 实际在库内 → 自动重判为库内样例
            ok = m["title"] in answer
            res["type"] = "library(auto)"
            res["pass"] = ok
            res["note"] = f"运行时发现《{m['title']}》在库内，已按库内判定：{'命中' if ok else '未命中'}"
            return res
        refused = bool(REFUSAL_RE.search(answer))
        leaked = (case["target"] in answer) and bool(ATTR_RE.search(answer))
        res["pass"] = (refused and not leaked)
        res["note"] = f"拒答={'是' if refused else '否'} / 越界泄漏={'是' if leaked else '否'}"
        res["_hallucinated"] = leaked
        res["_refused"] = refused

    elif t == "chat":
        mentioned = titles_in_answer(answer)
        res["pass"] = (len(mentioned) == 0)
        res["note"] = f"闲聊不应提电影，实际提到：{mentioned or '无'}"

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

    elif t == "neg_genre":
        excluded = genre_titles(case["exclude"])
        bad = set(titles_in_answer(answer)) & excluded
        res["pass"] = len(bad) == 0
        res["note"] = f"排除「{case['exclude']}」，误含：{list(bad) or '无'}"

    elif t == "fuzzy":
        m = match_movie(case["target"])
        ok = bool(m) and ((case["target"] in answer) or (m["title"] in answer))
        res["pass"] = ok
        res["note"] = f"模糊匹配→《{m['title'] if m else case['target']}》：{'命中' if ok else '未命中'}"

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
        answer = chat(case["query"], f"eval-{case['id']}")
        r = score(case, answer)
        results.append(r)

        if not quiet:
            tag = "✅通过" if r["pass"] else ("⚠️观察" if r["pass"] is None else "❌未过")
            print(f"[{case['id']}] {r['type']:>12} | {case['query']}")
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
        "样本总数": len(results),
        "有效评分样本": total_scored,
        "总通过率": f"{(total_pass / total_scored * 100):.1f}%" if total_scored else "N/A",
        "库外样本数": refusal_total,
        "拒答准确率": f"{refuse_acc * 100:.1f}%",
        "幻觉率_库外编造虚假事实": f"{hallu_rate * 100:.1f}%",
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
