import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import http from 'http'

// 复用后端连接（keep-alive），避免浏览器每请求一张海报 / 每次切分类，
// vite 都向后端 8000 开一条全新 TCP 连接 → 端口上堆积大量 TIME_WAIT，
// 频繁刷新时偶发 ECONNRESET。复用后 vite 只维持少量长连接。
// maxSockets 给足余量；maxFreeSockets 让空闲连接保活复用。
const backendAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 100,
  maxFreeSockets: 10,
})

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 开发期把 /api 代理到 FastAPI 后端，避免跨域
      // 注意：用 127.0.0.1 而非 localhost，避免 Node 把 localhost 解析成
      // IPv6 的 ::1 而 uvicorn 只监听 IPv4 的 127.0.0.1，导致 ECONNREFUSED
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        agent: backendAgent,
        // 只对 SSE 流式接口（agent 逐 token 推送）关闭缓冲 + 禁用缓存；
        // 其余 /api（海报 / 列表 / 详情等）放行后端自带的 Cache-Control，
        // 让浏览器缓存海报，切分类/翻页不再重复下载，连接数骤降。
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes, req) => {
            const url = (req && req.url) || ''
            if (url.includes('/agent/chat/stream')) {
              proxyRes.headers['Cache-Control'] = 'no-cache'
              proxyRes.headers['X-Accel-Buffering'] = 'no'
            }
          })
        },
      },
      // 图片等静态资源也代理到后端 /static（海报、头像等），同样复用连接
      '/static': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        agent: backendAgent,
      },
    }
  }
})
