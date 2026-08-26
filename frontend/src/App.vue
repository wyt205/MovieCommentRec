<script setup>
import NavBar from './components/NavBar.vue'
import AppFooter from './components/AppFooter.vue'
</script>

<template>
  <NavBar />
  <main class="container" style="padding-top:76px;">
    <router-view v-slot="{ Component }">
      <keep-alive include="AgentChat">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </keep-alive>
    </router-view>
  </main>
  <AppFooter />
</template>

<style>
.fade-enter-active,
.fade-leave-active { transition: opacity .15s ease; }
.fade-enter-from,
.fade-leave-to { opacity: 0; }
</style>
