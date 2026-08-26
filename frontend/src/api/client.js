import axios from 'axios'

// 基址用 /api，开发期由 vite 代理转发到后端 8000
const client = axios.create({ baseURL: '/api' })

export default client
