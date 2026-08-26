<script setup>
import { ref } from 'vue'
import { useRouter, useRoute, RouterLink } from 'vue-router'

const router = useRouter()
const route = useRoute()
const keyword = ref(route.query.q || '')

function onSearch() {
  const q = keyword.value.trim()
  // 跳到电影列表页，并把关键词放到 ?q=，列表页会读取
  router.push(q ? { path: '/', query: { q } } : { path: '/' })
}
</script>

<template>
  <header style="background:#fff;border-bottom:1px solid var(--border);position:fixed;top:0;left:0;right:0;z-index:100;">
    <div class="container between" style="padding-top:10px;padding-bottom:10px;">
      <div class="row" style="gap:16px;">
        <RouterLink to="/" style="font-weight:700;font-size:18px;color:var(--text);text-decoration:none;">
          🎬 llm-pro 影评
        </RouterLink>
        <nav class="row" style="gap:14px;">
          <RouterLink to="/" class="muted">电影</RouterLink>
          <RouterLink to="/reviews" class="muted">影评</RouterLink>
          <RouterLink to="/about" class="muted">关于</RouterLink>
          <RouterLink to="/agent" class="muted">智能助手</RouterLink>
        </nav>
      </div>

      <form class="row" @submit.prevent="onSearch" style="flex:1;max-width:360px;justify-content:flex-end;">
        <input
          v-model="keyword"
          class="input"
          style="flex:1;"
          type="search"
          placeholder="搜索电影…"
          aria-label="搜索电影"
        />
        <button class="btn btn-primary" type="submit">搜索</button>
      </form>
    </div>
  </header>
</template>

<style scoped>
</style>
