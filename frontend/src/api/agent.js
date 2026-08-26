import client from './client'

// 调用智能助手：POST /api/agent/chat
export function agentChat(data) {
  return client.post('/agent/chat', data)
}
