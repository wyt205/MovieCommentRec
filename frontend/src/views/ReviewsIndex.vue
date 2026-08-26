<script setup>
import { ref, onMounted } from 'vue'
import { listReviews } from '../api/reviews'
import ReviewCard from '../components/ReviewCard.vue'

const reviews = ref([])
const loading = ref(false)
const sort = ref('recent')

async function load() {
  loading.value = true
  try {
    const { data } = await listReviews({ limit: 100, sort: sort.value })
    reviews.value = data
  } finally {
    loading.value = false
  }
}
onMounted(load)
</script>

<template>
  <div>
    <div class="between mb">
      <h2 style="margin:0;">影评广场</h2>
      <select v-model="sort" class="select" @change="load">
        <option value="recent">最新发布</option>
        <option value="useful">最多有用</option>
      </select>
    </div>

    <div v-if="loading" class="spinner"></div>
    <div v-else-if="reviews.length === 0" class="empty">
      还没有影评。在启动器里跑 TMDb 爬虫即可拉取真实影评。
    </div>
    <div v-else>
      <ReviewCard v-for="r in reviews" :key="r.id" :review="r" />
    </div>
  </div>
</template>
