<script setup>
const stack = [
  ['前端', 'Vue 3 + Vue Router + Vite'],
  ['后端', 'FastAPI + SQLAlchemy + Pydantic'],
  ['数据库', 'MySQL 8（电影 / 影评 / 短评 / 评论者）'],
  ['数据源', 'TMDb 官方 API（公开、无反爬、支持中文）'],
  ['海报', '存入数据库 BLOB，经 /api/movies/{id}/poster 回传'],
  ['Agent', '已接入 LangChain 真·Function Calling 自主智能体（推荐 / 检索 / 闲聊 / 身份问答）']
]
</script>

<template>
  <div class="card" style="max-width:760px;margin:0 auto;">
    <h1 style="margin-top:0;">关于智影影评</h1>
    <p style="line-height:1.8;color:#343a40;">
      这是一个用于演示「电影影评 + 智能推荐 Agent」的站点。当前阶段先把影评浏览体验做完整：
      你可以浏览电影、查看详情与影评、搜索与筛选、阅读完整影评。下一步会接入大模型 Agent，
      让你用自然语言问「帮我找一部适合周末看的悬疑片」「这部和哪部最像」之类的问题。
    </p>

    <h2 class="section-title">技术栈</h2>
    <table class="stack-table">
      <tr v-for="[k, v] in stack" :key="k">
        <td class="k">{{ k }}</td>
        <td>{{ v }}</td>
      </tr>
    </table>

    <h2 class="section-title">数据来源</h2>
    <p class="muted text-sm" style="line-height:1.8;">
      电影与影评数据来自 <b>TMDb（The Movie Database）</b> 官方免费 API，仅用于学习演示。
      影评以英文为主，电影元信息支持中文。海报图片经后端存入数据库，前端无需直连图床即可显示。
    </p>

    <h2 class="section-title">免责声明</h2>
    <p class="muted text-sm" style="line-height:1.8;">
      本站点为个人学习项目，数据归原版权方所有；如涉及侵权请联系删除。Agent 相关功能仍在开发中。
    </p>
  </div>
</template>

<style scoped>
.stack-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.stack-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
.stack-table .k { color: var(--muted); white-space: nowrap; width: 90px; }
</style>
