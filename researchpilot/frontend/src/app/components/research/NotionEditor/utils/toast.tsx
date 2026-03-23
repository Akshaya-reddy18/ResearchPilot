import React from 'react'

type ToastOptions = { duration?: number }

export function showToast(message: string, type: 'success' | 'error' = 'success', options?: ToastOptions) {
  const container = document.createElement('div')
  container.className = `fixed right-6 bottom-6 z-50 max-w-xs p-3 rounded-md shadow-lg text-sm text-white ${type === 'success' ? 'bg-emerald-600' : 'bg-rose-600'}`
  container.style.opacity = '0'
  container.style.transition = 'opacity 200ms ease, transform 200ms ease'
  container.innerText = message
  document.body.appendChild(container)
  requestAnimationFrame(() => {
    container.style.opacity = '1'
    container.style.transform = 'translateY(0)'
  })
  const duration = options?.duration ?? 3000
  setTimeout(() => {
    container.style.opacity = '0'
    setTimeout(() => container.remove(), 250)
  }, duration)
}

export default showToast
