import client from './client'

export function listReviews(params) {
  return client.get('/reviews', { params })
}

export function getReview(id) {
  return client.get(`/reviews/${id}`)
}

// 用户发布评论（带 1-5 星打分）
export function createReview(payload) {
  return client.post('/reviews', payload)
}
