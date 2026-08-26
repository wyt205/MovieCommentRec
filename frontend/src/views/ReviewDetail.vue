<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRoute, RouterLink } from 'vue-router'
import { getReview } from '../api/reviews'

const route = useRoute()
const review = ref(null)
const loading = ref(true)

const labelColor = {
  '力荐': '#e03131', '推荐': '#f08c00', '还行': '#1971c2',
  '较差': '#868e96', '很差': '#adb5bd'
}

async function load() {
  loading.value = true
  try {
    const { data } = await getReview(route.params.id)
    review.value = data
  } finally {
    loading.value = false
  }
}
onMounted(load)
watch(() => route.params.id, load)
</script>

<template>
  <div v-if="loading" class="spinner"></div>

  <div v-else-if="review">
    <RouterLink to="/reviews" class="muted text-sm">← 返回影评广场</RouterLink>

    <article class="card mt">
      <div class="rv-head">
        <h1 style="margin:0;font-size:22px;">{{ review.title }}</h1>
        <span v-if="review.rating_label" class="badge" :style="{ background: labelColor[review.rating_label] || '#868e96' }">
          {{ review.rating_label }}
        </span>
      </div>

      <div class="rv-sub muted text-sm">
        <span v-if="review.reviewer_name">✍️ {{ review.reviewer_name }}</span>
        <span v-if="review.publish_date">{{ review.publish_date.slice(0, 10) }}</span>
        <span>👍 {{ review.useful_count }} · 💬 {{ review.comments_count }} · 👁 {{ review.views }}</span>
      </div>

      <RouterLink v-if="review.movie_title" :to="`/movie/${review.movie_id}`" class="source-tag" style="display:inline-block;margin-top:8px;">
        🎬 {{ review.movie_title }} →
      </RouterLink>

      <hr style="border:none;border-top:1px solid var(--border);margin:16px 0;" />

      <div class="rv-content">{{ review.content || review.summary }}</div>
    </article>
  </div>

  <div v-else class="empty">未找到该影评。</div>
</template>

<style scoped>
.rv-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.rv-sub { display: flex; gap: 14px; margin-top: 8px; flex-wrap: wrap; }
.rv-content { line-height: 1.9; color: #343a40; white-space: pre-wrap; font-size: 15px; }
</style>
