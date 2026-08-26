<script setup>
import { ref, nextTick } from 'vue'
import { agentChat } from '../api/agent'

// 每个页面会话一个 session_id，让后端记住多轮上下文
const sessionId = 'web-' + Math.random().toString(36).slice(2, 10)

const messages = ref([
  { role: 'bot', text: '你好！我是智影智能助手 🤖\n试着问我：「帮我找 2024 年评分最高的喜剧」「《星际穿越》讲了什么」「类似《盗梦空间》的电影」' }
])
const input = ref('')
const sending = ref(false)
const chatBox = ref(null)

async function send() {
  const text = input.value.trim()
  if (!text || sending.value) return
  input.value = ''
  messages.value.push({ role: 'user', text })
  sending.value = true
  await scrollDown()
  try {
    const { data } = await agentChat({ message: text, session_id: sessionId })
    messages.value.push({ role: 'bot', text: data.reply || '（助手没有返回内容）' })
  } catch (e) {
    const detail = e?.response?.data?.detail || e.message
    messages.value.push({ role: 'bot', text: `⚠️ 请求失败：${detail}` })
  } finally {
    sending.value = false
    await scrollDown()
  }
}

async function scrollDown() {
  await nextTick()
  if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
}
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

    <div class="chat card" ref="chatBox">
      <div
        v-for="(m, i) in messages"
        :key="i"
        class="bubble"
        :class="m.role"
      >{{ m.text }}</div>

      <div v-if="sending" class="bubble bot typing">正在思考…</div>
    </div>

    <div class="composer">
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
  </div>
</template>

<style scoped>
.agent-page { max-width: 760px; margin: 0 auto; }
.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
  background: #f1f3f5;
}
.chat {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  height: 60vh;
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
.composer {
  display: flex;
  gap: 10px;
  margin-top: 14px;
}
.composer .input { flex: 1; }
</style>
