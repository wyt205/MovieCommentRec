<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { listMovies, getStats, getGenres } from '../api/movies'
import MovieCard from '../components/MovieCard.vue'
import RankingPanel from '../components/RankingPanel.vue'

const route = useRoute()

const movies = ref([])
const stats = ref({ movies: 0, reviews: 0, avg_rating: null })
const genres = ref([])
const loading = ref(false)
const keyword = ref(route.query.q || '')
const activeGenre = ref('')         // 当前选中分类，'' = 全部
const gridSort = ref('rating')      // 右栏网格排序：release | popularity | rating
const rankingMode = ref('popularity') // 左栏排行方式：popularity | rating
const page = ref(1)
const pageSize = 20
const finished = ref(false)

async function loadGenres() {
  try {
    const { data } = await getGenres()
    genres.value = data
  } catch (e) { /* 分类拉取失败不影响列表 */ }
}

async function loadStats() {
  try {
    const { data } = await getStats()
    stats.value = data
  } catch (e) { /* 统计失败不影响列表 */ }
}

async function loadGrid(reset = false) {
  if (reset) { page.value = 1; movies.value = []; finished.value = false }
  loading.value = true
  try {
    const { data } = await listMovies({
      keyword: keyword.value.trim() || undefined,
      genre: activeGenre.value || undefined,
      sort: gridSort.value,
      skip: (page.value - 1) * pageSize,
      limit: pageSize
    })
    if (data.length < pageSize) finished.value = true
    movies.value.push(...data)
  } finally {
    loading.value = false
  }
}

function setGenre(g) {
  activeGenre.value = g
  loadGrid(true)   // 右栏网格变化
  // 左栏排行榜通过 RankingPanel 监听 activeGenre 自动刷新
}

function loadMore() {
  page.value += 1
  loadGrid(false)
}

onMounted(() => { loadGenres(); loadStats(); loadGrid(true) })

watch(() => route.query.q, (q) => {
  keyword.value = q || ''
  loadGrid(true)
})
</script>

<template>
  <div>
    <!-- 两栏：左排行 / 右 分类+排序+网格 -->
    <div class="layout">
      <aside class="col-left">
        <RankingPanel :genre="activeGenre" v-model:mode="rankingMode" />
      </aside>

      <div class="col-right">
        <!-- 分类标签（来自 /api/genres，严格按 TMDb 官方类型） -->
        <div class="genres-row">
          <span class="chip" :class="{ active: activeGenre === '' }" @click="setGenre('')">全部</span>
          <span
            v-for="g in genres"
            :key="g.id"
            class="chip"
            :class="{ active: activeGenre === g.name }"
            @click="setGenre(g.name)"
          >{{ g.name }}</span>
        </div>

        <!-- 工具栏：统计 + 排序下拉 -->
        <div class="toolbar between mt">
          <span class="muted text-sm">
            共 <b>{{ stats.movies }}</b> 部电影 · <b>{{ stats.reviews }}</b> 条影评
          </span>
          <div class="row">
            <span class="muted text-sm">排序</span>
            <select v-model="gridSort" class="select" @change="loadGrid(true)">
              <option value="release">上映时间</option>
              <option value="popularity">热度</option>
              <option value="rating">评分</option>
            </select>
          </div>
        </div>

        <!-- 电影网格 -->
        <div v-if="loading && movies.length === 0" class="spinner"></div>
        <div v-else-if="movies.length === 0" class="empty mt">
          没有匹配的电影。试试其他关键词，或先在启动器里跑 TMDb 爬虫拉取数据。
        </div>

        <div v-else class="grid mt">
          <MovieCard v-for="m in movies" :key="m.id" :movie="m" />
        </div>

        <div v-if="loading && movies.length" class="spinner"></div>
        <div v-if="!finished && movies.length" class="center mt">
          <button class="btn" :disabled="loading" @click="loadMore">加载更多</button>
        </div>
        <div v-else-if="finished && movies.length" class="center muted text-sm mt">
          — 已经到底啦 —
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 16px;
  margin-top: 16px;
  align-items: start;
}
.col-right { min-width: 0; }
.genres-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

/* 窄屏：单栏堆叠，排行榜移到上方 */
@media (max-width: 768px) {
  .layout { grid-template-columns: 1fr; }
  .rank-panel { position: static; }
}
</style>
