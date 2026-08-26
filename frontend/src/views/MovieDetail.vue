<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, RouterLink } from 'vue-router'
import { getMovie, listMovies } from '../api/movies'
import { listReviews, createReview } from '../api/reviews'
import StarRating from '../components/StarRating.vue'

const route = useRoute()
const movie = ref(null)
const reviews = ref([])
const similar = ref([])
const posterFailed = ref(false)
const loading = ref(true)

const labelColor = {
  '力荐': '#e03131', '推荐': '#f08c00', '还行': '#1971c2',
  '较差': '#868e96', '很差': '#adb5bd'
}

// 发布短评表单
const myRating = ref(0)
const myComment = ref('')
const submitting = ref(false)
const submitMsg = ref('')
const submitOk = ref(false)

async function load() {
  const id = route.params.id
  loading.value = true
  posterFailed.value = false
  submitMsg.value = ''
  try {
    const { data: m } = await getMovie(id)
    movie.value = m

    const [{ data: rv }] = await Promise.all([
      listReviews({ movie_id: id, limit: 50 })
    ])
    reviews.value = rv

    // 相似推荐：取第一个类型，拉同类型电影（排除自己），最多 5 部
    const g = (m.genres || '').split(/[/,，]/)[0]?.trim()
    if (g) {
      const { data } = await listMovies({ genre: g, limit: 6 })
      similar.value = data.filter(x => x.id !== m.id).slice(0, 5)
    } else {
      similar.value = []
    }
  } finally {
    loading.value = false
  }
}

onMounted(load)
watch(() => route.params.id, load)

// 演员只显示名称（不挂头像），与导演一致的纯文本展示
const castNames = computed(() => {
  const c = movie.value?.cast || []
  return c.map(x => x.name).filter(Boolean).join(' / ') || ''
})

// 评分分布（来自 movie.rating_distribution）
const dist = computed(() => movie.value?.rating_distribution || null)
function pctFor(star) {
  const d = dist.value
  if (!d || !d.total) return 0
  const cnt = (d.distribution && d.distribution[star]) || 0
  return Math.round((cnt / d.total) * 100)
}

// 上映日期 YYYY-MM-DD → YYYY年MM月DD日
function fmtDate(s) {
  if (!s) return ''
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s)
  if (m) return `${m[1]}年${m[2]}月${m[3]}日`
  return s
}

const posterUrl = computed(() => {
  if (posterFailed.value) return null
  return movie.value?.poster_url || null
})

async function submitReview() {
  if (!movie.value) return
  if (myRating.value < 1) {
    submitMsg.value = '请点击星星打分'
    submitOk.value = false
    return
  }
  if (!myComment.value.trim()) {
    submitMsg.value = '评论内容不能为空'
    submitOk.value = false
    return
  }
  submitting.value = true
  submitMsg.value = ''
  try {
    await createReview({
      movie_id: movie.value.id,
      rating: myRating.value,
      content: myComment.value.trim(),
    })
    submitOk.value = true
    submitMsg.value = '发布成功，感谢你的评价！'
    myRating.value = 0
    myComment.value = ''
    await load() // 刷新评论列表与评分分布
  } catch (e) {
    submitOk.value = false
    submitMsg.value = '发布失败：' + (e?.response?.data?.detail || e.message)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div v-if="loading" class="spinner"></div>

  <div v-else-if="movie" class="detail">
    <!-- 板块1：头部 -->
    <section class="movie-header card">
      <img
        v-if="posterUrl"
        :src="posterUrl"
        :alt="movie.title"
        class="poster"
        @error="posterFailed = true"
      />
      <div v-else class="poster fallback">{{ (movie.title || '?').slice(0, 1) }}</div>

      <!-- 中间：标题 + 基本信息 -->
      <div class="main-info">
        <h1 class="title">{{ movie.title }}</h1>
        <p class="original muted" v-if="movie.original_title && movie.original_title !== movie.title">
          {{ movie.original_title }}
        </p>

        <ul class="meta-list">
          <li><b>上映时间</b><span>{{ movie.release_date ? fmtDate(movie.release_date) : '暂无上映时间' }}</span></li>
          <li v-if="movie.country"><b>国家/地区</b><span>{{ movie.country }}</span></li>
          <li v-if="movie.genres">
            <b>分类</b>
            <span class="chips">
              <span v-for="g in movie.genres.split(/[/,，]/).filter(Boolean)" :key="g" class="chip">{{ g.trim() }}</span>
            </span>
          </li>
          <li v-if="movie.directors"><b>导演</b><span>{{ movie.directors }}</span></li>
          <li v-if="castNames"><b>演员</b><span>{{ castNames }}</span></li>
        </ul>
      </div>

      <!-- 右上角：评分 + 五星 + 评分分布 + 共计 -->
      <div class="score-panel">
        <div class="panel-score">
          <span class="score">{{ movie.rating ?? '—' }}</span>
          <StarRating :value="movie.rating ? movie.rating / 2 : 0" :size="22" readonly />
        </div>

        <div class="dist" v-if="dist && dist.total">
          <div v-for="s in [5, 4, 3, 2, 1]" :key="s" class="dist-row">
            <span class="dist-label">{{ s }}星</span>
            <span class="dist-track"><span class="dist-bar" :style="{ width: pctFor(s) + '%' }"></span></span>
            <span class="dist-pct">{{ pctFor(s) }}%</span>
          </div>
        </div>
        <div v-else class="dist-empty muted">
          TMDb 已有 <b>{{ movie.rating_count ?? 0 }}</b> 人评分<br />
          <span class="dist-tip">（暂无用户打分分布）</span>
        </div>

        <div class="dist-foot muted">共计 {{ movie.review_count }} 条评论</div>
      </div>
    </section>

    <!-- 板块2：剧情简介 -->
    <section v-if="movie.summary" class="section card">
      <h2 class="section-title">剧情简介</h2>
      <p class="summary">{{ movie.summary }}</p>
      <p class="tagline muted" v-if="movie.tagline">“{{ movie.tagline }}”</p>
    </section>

    <!-- 板块3：评论区 -->
    <section class="section card">
      <h2 class="section-title">评论区（{{ movie.review_count }}）</h2>

      <!-- 评论列表 / 空状态（在上方） -->
      <div v-if="reviews.length === 0" class="empty">暂无评论</div>
      <article v-for="r in reviews" :key="r.id" class="review-block">
        <div class="rv-head">
          <RouterLink :to="`/review/${r.id}`" class="rv-title">{{ r.title }}</RouterLink>
          <span v-if="r.rating_label" class="badge" :style="{ background: labelColor[r.rating_label] || '#868e96' }">
            {{ r.rating_label }}
          </span>
          <StarRating v-if="r.rating" :value="r.rating" :size="14" readonly />
        </div>
        <div class="rv-sub muted text-sm">
          <span v-if="r.reviewer_name">✍️ {{ r.reviewer_name }}</span>
          <span v-if="r.publish_date">{{ r.publish_date.slice(0, 10) }}</span>
        </div>
        <p class="rv-content">{{ r.content || r.summary }}</p>
        <div class="rv-foot muted text-sm">
          👍 {{ r.useful_count }} · 💬 {{ r.comments_count }} ·
          <RouterLink :to="`/review/${r.id}`">阅读全文 →</RouterLink>
        </div>
      </article>

      <!-- 发布评论（在下方） -->
      <div class="review-form">
        <div class="form-row">
          <StarRating :value="myRating" :size="26" @update:value="myRating = $event" />
          <input
            v-model="myComment"
            class="input comment-input"
            type="text"
            placeholder="写几句评价…"
            maxlength="500"
            @keyup.enter="submitReview"
          />
          <button class="btn btn-primary" :disabled="submitting" @click="submitReview">
            {{ submitting ? '发送中…' : '发送' }}
          </button>
        </div>
        <div v-if="submitMsg" class="form-msg" :class="submitOk ? 'ok' : 'err'">{{ submitMsg }}</div>
      </div>
    </section>

    <!-- 板块4：相似推荐 -->
    <section v-if="similar.length" class="section card">
      <h2 class="section-title">相似推荐</h2>
      <div class="grid">
        <RouterLink
          v-for="s in similar"
          :key="s.id"
          :to="`/movie/${s.id}`"
          class="movie-card"
        >
          <div class="poster-wrap">
            <img v-if="s.poster_url" :src="s.poster_url" :alt="s.title" class="poster" loading="lazy" />
            <div v-else class="poster fallback">{{ (s.title || '?').slice(0, 1) }}</div>
          </div>
          <div class="mc-info">
            <div class="mc-title">{{ s.title }} <span class="mc-year" v-if="s.year">{{ s.year }}</span></div>
            <div class="mc-meta muted text-sm">{{ s.genres || '未知类型' }}</div>
          </div>
        </RouterLink>
      </div>
    </section>
  </div>

  <div v-else class="empty">未找到该电影。</div>
</template>

<style scoped>
.detail {
  max-width: 1000px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 头部：三栏，三列等高 */
.movie-header {
  display: grid;
  grid-template-columns: 180px 1fr 260px;
  gap: 24px;
  align-items: stretch;
}
.poster {
  width: 100%; height: 100%; object-fit: cover; border-radius: 10px;
  background: #e9ecef; flex-shrink: 0;
}
.poster.fallback {
  display: flex; align-items: center; justify-content: center;
  height: 100%;
  font-size: 80px; font-weight: 700; color: #fff;
  background: linear-gradient(135deg, #495057, #868e96);
}

.main-info { min-width: 0; }
.title { margin: 0; font-size: 26px; line-height: 1.25; }
.original { margin: 4px 0 0; font-style: italic; }

.meta-list { list-style: none; padding: 0; margin: 8px 0 0; }
.meta-list li {
  display: flex; gap: 10px; padding: 5px 0; font-size: 14px; border-top: 1px solid #f1f3f5;
}
.meta-list li:first-child { border-top: none; }
.meta-list b { color: var(--muted); width: 72px; flex-shrink: 0; font-weight: 600; }
.meta-list span { min-width: 0; }

.chips { display: inline-flex; flex-wrap: wrap; gap: 6px; }
.chip {
  background: #f1f3f5; color: #495057; border-radius: 999px;
  padding: 2px 10px; font-size: 12px;
}

/* 右上角评分面板：整体垂直居中 */
.score-panel {
  border-left: 1px solid #f1f3f5;
  padding-left: 24px;
  min-width: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 16px;
}
/* 评分数字居左、五星居右，共占一行 */
.panel-score {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.panel-score .score { font-size: 30px; font-weight: 800; color: var(--primary); line-height: 1; }

.dist {
  font-size: 13px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.dist-row { display: flex; align-items: center; gap: 8px; }
.dist-label { width: 30px; color: var(--muted); flex-shrink: 0; }
.dist-track { flex: 1; height: 12px; background: #eef0f2; border-radius: 6px; overflow: hidden; }
.dist-bar {
  display: block; height: 100%;
  background: linear-gradient(90deg, #ffd43b, #fab005);
  border-radius: 6px; transition: width .4s ease;
}
.dist-pct { width: 40px; text-align: right; color: var(--muted); flex-shrink: 0; }
.dist-foot { font-size: 13px; }
.dist-empty {
  font-size: 13px;
  line-height: 1.7;
}
.dist-empty b { color: var(--primary); }
.dist-tip { font-size: 12px; }

/* 通用区块 */
.section { padding: 18px 20px; }
.section-title { margin: 0 0 14px; font-size: 18px; }
.summary { line-height: 1.9; margin: 0; color: #343a40; white-space: pre-wrap; }
.tagline { font-style: italic; margin: 10px 0 0; }

/* 评论区 */
.review-form {
  background: #f8f9fa; border: 1px solid #f1f3f5; border-radius: 10px;
  padding: 12px 14px; margin-top: 18px;
}
.form-row {
  display: flex; gap: 12px; align-items: center;
}
.comment-input {
  flex: 1;
  height: 40px;
  padding: 0 12px;
  box-sizing: border-box;
}
.form-msg { font-size: 13px; margin-top: 8px; }
.form-msg.ok { color: #2f9e44; }
.form-msg.err { color: #e03131; }

.review-block { border-top: 1px solid #f1f3f5; padding: 14px 0; }
.review-block:first-of-type { border-top: none; padding-top: 0; }
.rv-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.rv-title { font-weight: 600; color: var(--text); text-decoration: none; font-size: 16px; }
.rv-title:hover { color: var(--primary); }
.rv-sub { display: flex; gap: 12px; margin-top: 4px; flex-wrap: wrap; }
.rv-content { line-height: 1.8; color: #343a40; margin: 8px 0; white-space: pre-wrap; }
.rv-foot { margin-top: 6px; }

.empty {
  text-align: center; color: var(--muted); padding: 24px 0; font-size: 14px;
}

/* 相似推荐：最多 5 个，平铺占满一行 */
.grid {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px;
}
.movie-card { text-decoration: none; color: inherit; }
.poster-wrap { aspect-ratio: 2 / 3; border-radius: 8px; overflow: hidden; background: #e9ecef; }
.grid .poster { width: 100%; height: 100%; object-fit: cover; display: block; }
.grid .poster.fallback {
  display: flex; align-items: center; justify-content: center;
  font-size: 40px; font-weight: 700; color: #fff;
  background: linear-gradient(135deg, #495057, #868e96);
}
.mc-info { padding: 6px 2px; }
.mc-title { font-size: 14px; font-weight: 600; }
.mc-year { color: var(--muted); font-weight: 400; }
.mc-meta { margin-top: 2px; }

@media (max-width: 900px) {
  .grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 560px) {
  .grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 820px) {
  .movie-header { grid-template-columns: 140px 1fr; }
  .score-panel {
    grid-column: 1 / -1;
    border-left: none;
    border-top: 1px solid #f1f3f5;
    padding-left: 0;
    padding-top: 18px;
  }
}

@media (max-width: 640px) {
  .movie-header { grid-template-columns: 1fr; text-align: center; }
  .poster { width: 160px; height: 226px; margin: 0 auto; }
  .meta-list li { justify-content: center; }
  .meta-list b { width: auto; }
  .panel-score { align-items: center; }
  .form-row { flex-wrap: wrap; }
  .comment-input { width: 100%; }
}
</style>
