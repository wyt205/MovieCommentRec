<script>
export default { name: 'AgentChat' }
</script>

<script setup>
import { ref, reactive, nextTick, onMounted } from 'vue'

const WELCOME = {
  role: 'bot',
  text: '你好！我是智影智能助手 🤖\n试着问我：「帮我找 2024 年评分最高的喜剧」「《星际穿越》讲了什么」「类似《盗梦空间》的电影」',
}

const sessions = ref([])          // 左侧会话列表 [{session_id, title, count, updated_at}]
const currentSid = ref(null)      // 当前选中的会话 id（由后端分配，不再随机）
const messages = ref([])          // 当前会话的对话消息
const input = ref('')
const sending = ref(false)
const chatBox = ref(null)
const autonomous = ref(false)   // 自主 Agent 模式开关：开=LLM 自主 function calling 为主路径

async function api(path, opts) {
  const resp = await fetch('/api' + path, opts)
  if (!resp.ok) throw new Error('HTTP ' + resp.status)
  return resp
}

async function loadSessions() {
  try {
    const resp = await api('/sessions')
    sessions.value = await resp.json()
  } catch (e) {
    sessions.value = []
  }
}

async function selectSession(sid) {
  currentSid.value = sid
  try {
    const resp = await api('/sessions/' + sid)
    const data = await resp.json()
    const msgs = (data.messages || []).map((m) => ({
      role: m.role === 'user' ? 'user' : 'bot',
      text: m.text,
    }))
    messages.value = msgs.length ? msgs : [{ ...WELCOME }]
  } catch (e) {
    messages.value = [{ ...WELCOME }]
  }
  await scrollDown()
}

async function newSession() {
  try {
    const resp = await api('/sessions', { method: 'POST' })
    const data = await resp.json()
    sessions.value.unshift({ session_id: data.session_id, title: data.title, count: 0 })
    currentSid.value = data.session_id
    messages.value = [{ ...WELCOME }]
  } catch (e) {
    currentSid.value = null
    messages.value = [{ ...WELCOME }]
  }
}

async function deleteSession(sid) {
  try {
    await api('/sessions/' + sid, { method: 'DELETE' })
  } catch (e) {
    /* ignore */
  }
  sessions.value = sessions.value.filter((s) => s.session_id !== sid)
  if (currentSid.value === sid) {
    if (sessions.value.length) await selectSession(sessions.value[0].session_id)
    else await newSession()
  }
}

// 流式请求【直连后端】，绕过 Vite 开发代理的 SSE 缓冲（http-proxy 会整段缓冲、导致看不到逐字）。
// 后端已开启 CORS(allow_origins=*)，且现在启动器能可靠拉起后端，直连安全。
// 非流式请求（会话列表等）仍走相对路径 /api（经 Vite 代理）。
// 想换后端地址可在 frontend/.env 设 VITE_BACKEND_URL=http://127.0.0.1:8000
const STREAM_BASE = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'

async function send() {
  const text = input.value.trim()
  if (!text || sending.value || !currentSid.value) return
  input.value = ''
  messages.value.push({ role: 'user', text })
  // 必须用 reactive 包裹：push 进 ref([]) 后，数组里的元素是 reactive 代理，
  // 但此处闭包引用的 bot 仍是「原始对象」。若直接改原始对象（bot.text += …），
  // 不会经过 proxy 的 set trap → 不触发响应式更新 → 模板一直停在初始「思考中」，
  // 直到 finally 里 sending.value=false 触发整组件重渲染才一次性吐出全文。
  // 用 reactive() 让 bot 本身就是代理，后续 bot.thinking/bot.text 修改才会逐字触发重渲染。
  const bot = reactive({ role: 'bot', text: '', thinking: true, waiting: false })
  messages.value.push(bot)
  sending.value = true
  await scrollDown()
  // 若 2.5s 内未收到首个 token（后端正在做 agentic 循环 / 查电影库），给出更具体的等待提示，
  // 避免用户误以为「卡死 / 没流式」
  let gotFirst = false
  // 注意：等待提示只能写进独立的 waiting 标志，绝不能写进 bot.text——
  // 否则真实 token 会用 += 追加在提示语后面，导致「正在检索电影库…」混进答案开头。
  const waitTimer = setTimeout(() => {
    if (!gotFirst && bot.thinking) bot.waiting = true
  }, 1500)
  try {
    const resp = await fetch(`${STREAM_BASE}/api/agent/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: currentSid.value }),
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      bot.thinking = false
      bot.text = `⚠️ 请求失败：${err.detail || resp.status}`
      return
    }
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      gotFirst = true
      clearTimeout(waitTimer)
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const raw = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        let event = 'message'
        let data = ''
        for (const line of raw.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) data += line.slice(5).trim()
        }
        if (!data) continue
        if (event === 'status') {
          bot.waiting = true
        } else if (event === 'done') {
          bot.thinking = false
          try { bot.meta = JSON.parse(data) } catch { /* ignore */ }
        } else if (event === 'error') {
          bot.thinking = false
          const d = JSON.parse(data)
          bot.text += `\n⚠️ ${d.detail || '出错了'}`
        } else {
          bot.thinking = false
          bot.text += JSON.parse(data)
          await scrollDown()
        }
      }
    }
    bot.thinking = false
    clearTimeout(waitTimer)
    await loadSessions() // 刷新左侧列表（标题/顺序可能已更新）
  } catch (e) {
    bot.thinking = false
    clearTimeout(waitTimer)
    bot.text = `⚠️ 连接后端失败：${e?.message || e}（请确认启动器已「启动后端」）`
  } finally {
    sending.value = false
    await scrollDown()
  }
}

async function scrollDown() {
  await nextTick()
  if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
}

// —— 自主 Agent 模式开关（读/写后端运行时状态）——
async function loadMode() {
  try {
    const resp = await api('/agent/mode')
    const d = await resp.json()
    autonomous.value = !!d.autonomous
  } catch (e) {
    /* 后端未起时保持默认 false */
  }
}

async function toggleAutonomous() {
  const next = !autonomous.value
  try {
    const resp = await api('/agent/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ autonomous: next }),
    })
    const d = await resp.json()
    autonomous.value = !!d.autonomous
  } catch (e) {
    /* 失败时不切换，避免界面状态与后端不一致 */
  }
}

onMounted(async () => {
  await loadSessions()
  await loadMode()
  if (sessions.value.length) await selectSession(sessions.value[0].session_id)
  else await newSession()
})
</script>

<template>
  <div class="agent-page">
    <div class="agent-head between">
      <div class="row" style="gap:10px;">
        <h2 style="margin:0;">🤖 智能对话助手</h2>
        <span class="tag">Beta</span>
      </div>
      <span class="muted text-sm">基于你的影评数据库 · LangChain + function calling</span>
    </div>

    <div class="layout">
      <aside class="sidebar card">
        <button class="btn btn-primary new-btn" @click="newSession">+ 新建对话</button>
        <div class="sess-list">
          <div
            v-for="s in sessions"
            :key="s.session_id"
            class="sess-item"
            :class="{ active: s.session_id === currentSid }"
            @click="selectSession(s.session_id)"
          >
            <div class="sess-title">{{ s.title }}</div>
            <div class="sess-meta">{{ s.count }} 条</div>
            <button class="del" title="删除对话" @click.stop="deleteSession(s.session_id)">×</button>
          </div>
          <div v-if="!sessions.length" class="sess-empty">还没有对话，点上方新建</div>
        </div>
      </aside>

      <section class="chat-area">
        <div class="chat card" ref="chatBox">
          <div
            v-for="(m, i) in messages"
            :key="i"
            class="bubble"
            :class="[m.role, { typing: m.thinking && !m.text }]"
          >{{ m.thinking && !m.text ? (m.waiting ? '正在检索电影库…' : '正在思考…') : m.text }}</div>
        </div>

        <div class="composer">
          <button
            class="btn mode-toggle"
            :class="{ on: autonomous }"
            :title="autonomous ? '当前：自主 Agent 模式（LLM 自主选工具 + 护栏）' : '当前：规则路由模式（默认，可靠零幻觉）'"
            @click="toggleAutonomous"
          >🤖 自主模式: {{ autonomous ? '开' : '关' }}</button>
          <input
            class="input"
            v-model="input"
            placeholder="问我电影、影评、推荐…（Enter 发送）"
            :disabled="sending"
            @keyup.enter="send"
            aria-label="消息输入框"
          />
          <button class="btn btn-primary" :disabled="sending" @click="send">
            {{ sending ? '发送中…' : '发送' }}
          </button>
        </div>
        <p v-if="autonomous" class="mode-hint">🤖 自主 Agent 模式已开启：本次对话由 LLM 通过 function calling 自主决定调用哪个工具，护栏会强制它先查库再回答——可在管理端「日志」看到护栏列变「是」。</p>
      </section>
    </div>
  </div>
</template>

<style scoped>
.agent-page { max-width: 1100px; margin: 0 auto; }
.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
  background: #f1f3f5;
}
.layout { display: flex; gap: 14px; align-items: stretch; }
.sidebar {
  width: 240px;
  flex: 0 0 240px;
  display: flex;
  flex-direction: column;
  padding: 12px;
  height: 64vh;
}
.new-btn { width: 100%; margin-bottom: 10px; }
.sess-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
.sess-item {
  position: relative;
  padding: 10px 28px 10px 12px;
  border-radius: 10px;
  background: #f7f8fa;
  border: 1px solid transparent;
  cursor: pointer;
  transition: background .15s, border-color .15s;
}
.sess-item:hover { background: #eef0f4; }
.sess-item.active { background: #ece9ff; border-color: var(--primary, #6d5efc); }
.sess-title {
  font-size: 14px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.sess-meta { font-size: 12px; color: var(--muted); margin-top: 2px; }
.del {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 20px;
  height: 20px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: var(--muted);
  font-size: 16px;
  line-height: 1;
  cursor: pointer;
}
.del:hover { background: #ffd5d5; color: #c0392b; }
.sess-empty { color: var(--muted); font-size: 13px; padding: 12px; text-align: center; }

.chat-area { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.chat {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  height: 64vh;
  overflow-y: auto;
}
.bubble {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 14px;
  line-height: 1.6;
  font-size: 14px;
  white-space: pre-wrap;
}
.bubble.bot {
  align-self: flex-start;
  background: var(--surface, #f2f3f7);
  border: 1px solid var(--border, #e5e7eb);
  border-bottom-left-radius: 4px;
}
.bubble.user {
  align-self: flex-end;
  background: var(--primary, #6d5efc);
  color: #fff;
  border-bottom-right-radius: 4px;
}
.bubble.typing { opacity: .7; font-style: italic; }
.composer { display: flex; gap: 10px; margin-top: 14px; }
.composer .input { flex: 1; }
.mode-toggle {
  white-space: nowrap;
  background: #f1f3f5;
  color: var(--muted, #555);
  border: 1px solid var(--border, #e5e7eb);
}
.mode-toggle.on {
  background: var(--primary, #6d5efc);
  color: #fff;
  border-color: var(--primary, #6d5efc);
}
.mode-hint {
  margin: 8px 2px 0;
  font-size: 12px;
  color: var(--primary, #6d5efc);
  line-height: 1.5;
}

@media (max-width: 720px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; flex: none; height: auto; max-height: 30vh; }
}
</style>
