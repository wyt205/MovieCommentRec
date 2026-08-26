import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 开发期把 /api 代理到 FastAPI 后端，避免跨域
      // 注意：用 127.0.0.1 而非 localhost，避免 Node 把 localhost 解析成
      // IPv6 的 ::1 而 uvicorn 只监听 IPv4 的 127.0.0.1，导致 ECONNREFUSED
      '/api': 'http://127.0.0.1:8000',
      // 图片等静态资源也代理到后端 /static（海报、头像等）
      '/static': 'http://127.0.0.1:8000'
    }
  }
})
