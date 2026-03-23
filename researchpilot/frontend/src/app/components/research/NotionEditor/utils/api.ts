import { updateDocument } from '../../../../services/api'

// Export endpoints live at /api/export on the backend (not under /api/v1)
const BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export async function saveDraft(docId: string | number, html: string) {
  try {
    await updateDocument(Number(docId), html)
  } catch (e) {
    console.error('saveDraft failed', e)
    throw e
  }
}

async function fetchWithAuth(url: string, opts: RequestInit = {}) {
  const token = localStorage.getItem('token')
  const headers: HeadersInit = {
    ...(opts.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }

  const res = await fetch(url, { ...opts, headers })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || 'Request failed')
  }
  return res
}

export async function exportPDF(docId: string | number) {
  const url = `${BASE}/export/doc/${docId}/pdf`
  const res = await fetchWithAuth(url, { method: 'GET' })
  return res.blob()
}

export async function exportDOCX(docId: string | number) {
  const url = `${BASE}/export/doc/${docId}/docx`
  const res = await fetchWithAuth(url, { method: 'GET' })
  return res.blob()
}
