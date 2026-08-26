<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  // 当前分值（支持小数，如 4.3 表示 4.3/5）
  value: { type: Number, default: 0 },
  // 满分星数
  max: { type: Number, default: 5 },
  // 只读模式（仅展示，不可点击）
  readonly: { type: Boolean, default: true },
  // 星星字号(px)
  size: { type: Number, default: 20 },
})
const emit = defineEmits(['update:value', 'change'])

const hover = ref(0)
// 展示值：悬停时跟随鼠标，否则用真实 value
const shown = computed(() => hover.value || props.value)
// 第 i 颗星(1-based)的填充比例：clamp(value-(i-1), 0, 1)
function frac(i) {
  const v = shown.value - (i - 1)
  return Math.max(0, Math.min(1, v))
}
function pick(i) {
  if (props.readonly) return
  emit('update:value', i)
  emit('change', i)
}
function over(i) {
  if (!props.readonly) hover.value = i
}
function leave() {
  hover.value = 0
}
</script>

<template>
  <span
    class="stars"
    :class="{ ro: readonly }"
    :style="{ fontSize: size + 'px' }"
    @mouseleave="leave"
  >
    <span
      v-for="i in max"
      :key="i"
      class="star"
      @click="pick(i)"
      @mouseenter="over(i)"
    >
      <span class="base">★</span>
      <span class="fill" :style="{ width: (frac(i) * 100) + '%' }">★</span>
    </span>
  </span>
</template>

<style scoped>
.stars {
  display: inline-flex;
  line-height: 1;
  vertical-align: middle;
  cursor: default;
  user-select: none;
}
.stars:not(.ro) {
  cursor: pointer;
}
.star {
  position: relative;
  display: inline-block;
  color: #dfe2e6; /* 底色灰星 */
  transition: transform 0.05s;
}
.stars:not(.ro) .star:hover {
  transform: scale(1.15);
}
.star .fill {
  position: absolute;
  left: 0;
  top: 0;
  overflow: hidden;
  white-space: nowrap;
  color: #fab005; /* 金色填充 */
}
</style>
