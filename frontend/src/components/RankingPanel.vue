<script setup>
import { ref, onMounted, watch } from 'vue'
import { listMovies } from '../api/movies'

const props = defineProps({
  genre: { type: String, default: '' },      // 当前选中的分类（'' = 全部）
  mode: { type: String, default: 'popularity' } // popularity | rating
})
const emit = defineEmits(['update:mode'])

const list = ref([])
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    const { data } = await listMovies({
      genre: props.genre || undefined,
      sort: props.mode,
      limit: 10
    })
    list.value = data
  } catch (e) {
    list.value = []
  } finally {
    loading.value = false
  }
}

onMounted(load)
// 分类或排序方式变化时，自动重新拉取对应排行
watch(() => [props.genre, props.mode], load)

function rankClass(i) {
  return i < 3 ? `top${i + 1}` : ''
}
function metric(m) {
  return props.mode === 'popularity'
    ? `热度 ${(Number(m.popularity) || 0).toFixed(0)}`
    : `评分 ${m.rating ?? '—'}`
}
</script>

<template>
  <section class="card rank-panel">
    <div class="rank-head">
      <h3 class="rank-title-h">排行榜</h3>
      <div class="seg">
        <button :class="{ active: mode === 'popularity' }" @click="emit('update:mode', 'popularity')">热度</button>
        <button :class="{ active: mode === 'rating' }" @click="emit('update:mode', 'rating')">评分</button>
      </div>
    </div>
    <p class="rank-sub muted text-sm">
      {{ genre ? `「${genre}」分类` : '全部分类' }} · {{ mode === 'popularity' ? '按热度' : '按评分' }}
    </p>

    <ol class="rank-list" v-if="list.length">
      <li v-for="(m, i) in list" :key="m.id" class="rank-item">
        <RouterLink :to="`/movie/${m.id}`" class="rank-link">
          <span class="rank-no" :class="rankClass(i)">{{ i + 1 }}</span>
          <span class="rank-poster">
            <img v-if="m.poster_url" :src="m.poster_url" :alt="m.title" loading="lazy" @error="e => e.target.style.display='none'" />
          </span>
          <span class="rank-meta">
            <span class="rank-name">{{ m.title }}</span>
            <span class="rank-metric muted text-sm">{{ metric(m) }}</span>
          </span>
        </RouterLink>
      </li>
    </ol>

    <div v-else-if="loading" class="spinner"></div>
    <div v-else class="muted text-sm rank-empty">暂无数据</div>
  </section>
</template>

<style scoped>
.rank-panel { position: sticky; top: 16px; }
.rank-head {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.rank-title-h { margin: 0; font-size: 16px; font-weight: 700; }
.rank-sub { margin: 4px 0 12px; }
.rank-list { list-style: none; margin: 0; padding: 0; }
.rank-item { margin-bottom: 4px; }
.rank-link {
  display: flex; align-items: center; gap: 10px;
  padding: 6px; border-radius: 8px; color: var(--text); text-decoration: none;
}
.rank-link:hover { background: var(--surface-2); }
.rank-no {
  flex: 0 0 22px; height: 22px; line-height: 22px; text-align: center;
  border-radius: 6px; background: var(--surface-2); color: var(--muted);
  font-weight: 700; font-size: 13px;
}
.rank-no.top1 { background: #ffe066; color: #664d03; }
.rank-no.top2 { background: #e9ecef; color: #495057; }
.rank-no.top3 { background: #fcd5b5; color: #7a4a1e; }
.rank-poster {
  flex: 0 0 34px; height: 48px; border-radius: 4px; overflow: hidden;
  background: #e9ecef; display: flex; align-items: center; justify-content: center;
}
.rank-poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.rank-meta { display: flex; flex-direction: column; min-width: 0; }
.rank-name {
  font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rank-metric { white-space: nowrap; }
.rank-empty { padding: 24px 0; text-align: center; }

/* 分段切换按钮 */
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button {
  border: none; background: #fff; color: var(--muted); padding: 5px 12px;
  font-size: 13px; cursor: pointer;
}
.seg button + button { border-left: 1px solid var(--border); }
.seg button.active { background: var(--primary); color: #fff; }
</style>
