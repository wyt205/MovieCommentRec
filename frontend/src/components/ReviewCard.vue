<script setup>
import { RouterLink } from 'vue-router'

const props = defineProps({ review: Object, link: { type: Boolean, default: true } })

const labelColor = {
  '力荐': '#e03131',
  '推荐': '#f08c00',
  '还行': '#1971c2',
  '较差': '#868e96',
  '很差': '#adb5bd'
}
</script>

<template>
  <div class="review-card">
    <div class="rv-head">
      <RouterLink v-if="link" :to="`/review/${review.id}`" class="rv-title">
        {{ review.title }}
      </RouterLink>
      <span v-else class="rv-title">{{ review.title }}</span>

      <span
        v-if="review.rating_label"
        class="badge"
        :style="{ background: labelColor[review.rating_label] || '#868e96' }"
      >{{ review.rating_label }}</span>
    </div>

    <div class="rv-sub muted text-sm" v-if="review.movie_title || review.publish_date">
      <RouterLink v-if="review.movie_title" :to="`/movie/${review.movie_id}`" class="rv-movie">
        🎬 {{ review.movie_title }}
      </RouterLink>
      <span v-if="review.publish_date">{{ review.publish_date.slice(0, 10) }}</span>
    </div>

    <div class="rv-summary">{{ review.content || review.summary }}</div>

    <div class="rv-foot muted text-sm">
      👍 {{ review.useful_count }} · 💬 {{ review.comments_count }}
    </div>
  </div>
</template>

<style scoped>
.review-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px;
  margin-bottom: 12px;
}
.rv-head { display: flex; align-items: center; gap: 8px; }
.rv-title {
  font-weight: 600;
  color: var(--text);
  text-decoration: none;
}
.rv-title:hover { color: var(--primary); }
.rv-sub {
  display: flex;
  gap: 12px;
  margin-top: 4px;
  flex-wrap: wrap;
}
.rv-movie { color: var(--primary); }
.rv-summary {
  color: #343a40;
  font-size: 14px;
  margin-top: 8px;
  line-height: 1.7;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.rv-foot { margin-top: 8px; }
</style>
