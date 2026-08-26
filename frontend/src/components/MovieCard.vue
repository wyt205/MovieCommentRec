<script setup>
import { ref, computed } from 'vue'
const props = defineProps({ movie: Object })
const posterFailed = ref(false)

// 兼容新旧分隔符（"动作 / 科幻" 或 "动作, 科幻"），统一显示为 "动作 / 科幻"
const genresText = computed(() => {
  const g = props.movie?.genres
  if (!g) return '未知类型'
  return g.split(/[,，/]/).map(s => s.trim()).filter(Boolean).join(' / ')
})
</script>

<template>
  <RouterLink :to="`/movie/${movie.id}`" class="movie-card">
    <div class="poster-wrap">
      <img
        v-if="movie.poster_url && !posterFailed"
        :src="movie.poster_url"
        :alt="movie.title"
        class="poster"
        loading="lazy"
        @error="posterFailed = true"
      />
      <div v-else class="poster fallback">{{ (movie.title || '?').slice(0, 1) }}</div>
      <span v-if="movie.rating" class="badge badge-rating score">{{ movie.rating }}</span>
    </div>
    <div class="mc-info">
      <div class="mc-title">
        {{ movie.title }}
        <span class="mc-year" v-if="movie.year">{{ movie.year }}</span>
      </div>
      <div class="mc-meta muted text-sm">{{ genresText }}</div>
      <div class="mc-meta muted text-sm" v-if="movie.rating_count">
        {{ movie.rating_count.toLocaleString() }} 人评分
      </div>
    </div>
  </RouterLink>
</template>

<style scoped>
.movie-card {
  display: block;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  color: var(--text);
  text-decoration: none;
  transition: .15s;
}
.movie-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, .08);
}
.poster-wrap { position: relative; }
.poster {
  width: 100%;
  aspect-ratio: 2 / 3;
  object-fit: cover;
  display: block;
  background: #e9ecef;
}
.poster.fallback {
  width: 100%;
  aspect-ratio: 2 / 3;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 48px;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(135deg, #495057, #868e96);
}
.score {
  position: absolute;
  right: 8px;
  bottom: 8px;
  font-size: 13px;
}
.mc-info { padding: 8px 10px 10px; }
.mc-title {
  font-weight: 600;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.mc-year { color: var(--muted); font-weight: 400; }
.mc-meta {
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
