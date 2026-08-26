import { createRouter, createWebHistory } from 'vue-router'
import MovieList from '../views/MovieList.vue'
import MovieDetail from '../views/MovieDetail.vue'
import ReviewDetail from '../views/ReviewDetail.vue'
import ReviewsIndex from '../views/ReviewsIndex.vue'
import About from '../views/About.vue'
import AgentChat from '../views/AgentChat.vue'

const routes = [
  { path: '/', name: 'movies', component: MovieList },
  { path: '/movie/:id', name: 'movie-detail', component: MovieDetail, props: true },
  { path: '/review/:id', name: 'review-detail', component: ReviewDetail, props: true },
  { path: '/reviews', name: 'reviews', component: ReviewsIndex },
  { path: '/about', name: 'about', component: About },
  { path: '/agent', name: 'agent', component: AgentChat }
]

export default createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() { return { top: 0 } }
})
