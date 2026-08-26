<script setup>
const year = new Date().getFullYear()
</script>

<template>
  <footer style="border-top:1px solid var(--border);background:#fff;margin-top:40px;">
    <div class="container between" style="padding-top:18px;padding-bottom:18px;">
      <div class="muted text-sm">
        🎬 智影影评 · 一个用于演示 Agent + 推荐的电影影评站
      </div>
      <div class="row text-sm">
        <RouterLink to="/">电影</RouterLink>
        <RouterLink to="/reviews">影评</RouterLink>
        <RouterLink to="/about">关于</RouterLink>
        <span class="muted">© {{ year }}</span>
      </div>
    </div>
  </footer>
</template>
