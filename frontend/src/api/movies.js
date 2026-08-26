import client from './client'

export function listMovies(params) {
  return client.get('/movies', { params })
}

export function getMovie(id) {
  return client.get(`/movies/${id}`)
}

export function getStats() {
  return client.get('/movies/stats')
}

// 分类目录（TMDb 官方 19 个类型），作为分类标签的单一数据源
export function getGenres() {
  return client.get('/genres')
}
