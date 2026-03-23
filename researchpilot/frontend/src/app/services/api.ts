
/**
 * API Service for ResearchPilot Backend
 * Handles all HTTP requests to the FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ================= TYPES =================

export interface ChatRequest {
  query: string;
  use_context?: boolean;
  top_k?: number;
}

export interface ChatResponse {
  query: string;
  analysis: string | null;
  source_chunks_used: number;
  top_k?: number;
  model?: string;
  error?: string;
}

export interface SearchResult {
  rank: number;
  document: string;
  similarity: number;
  metadata: Record<string, any>;
}

export interface SearchResponse {
  status: string;
  query: string;
  results_count: number;
  results: SearchResult[];
}

export interface IngestionResponse {
  status: string;
  message: string;
  documents_ingested: number;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
}

export interface CollectionStats {
  status: string;
  data: {
    collection_name: string;
    document_count: number;
    embedding_model: number;
    db_path: string;
  };
}

// ================= AUTH HELPERS =================

export function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("user_email");
}

export function isAuthenticated(): boolean {
  return !!localStorage.getItem("token");
}

// ================= CORE REQUEST =================

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders: HeadersInit = {};
  if (!(options.body instanceof FormData)) {
    defaultHeaders['Content-Type'] = 'application/json';
  }

  const token = localStorage.getItem("token");

  try {
    console.log('API Request:', { url, method: options.method || 'GET' });

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
    });

    console.log('API Response:', response.status);

    // 🔥 AUTO LOGOUT ON 401
    if (response.status === 401) {
      console.warn("Unauthorized - logging out");

      logout();
      window.location.href = "/login";

      throw new Error("Session expired. Please login again.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        detail: response.statusText,
      }));

      let errorMessage = errorData.detail || response.statusText;

      // Handle nested API errors (Groq etc.)
      if (typeof errorMessage === 'string' && errorMessage.includes('error')) {
        try {
          const parsed = JSON.parse(
            errorMessage.replace(/^Error code: \d+ - /, '')
          );
          if (parsed.error?.message) {
            errorMessage = parsed.error.message;
          }
        } catch {}
      }

      throw new Error(errorMessage || `HTTP ${response.status}`);
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('API Request Failed:', error);

    if (error instanceof Error) {
      throw error;
    }

    throw new Error('Network error occurred');
  }
}

// ================= AUTH =================

export async function signup(
  email: string,
  password: string,
  full_name?: string
): Promise<any> {
  return apiRequest<any>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, full_name }),
  });
}

export async function login(
  email: string,
  password: string
): Promise<AuthTokenResponse> {
  const res = await apiRequest<AuthTokenResponse>('/api/auth/token', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });

  // 🔥 STORE TOKEN
  localStorage.setItem("token", res.access_token);
  localStorage.setItem("user_email", email);

  return res;
}

// ================= CHAT =================

export async function chat(request: ChatRequest): Promise<ChatResponse> {
  return apiRequest<ChatResponse>('/api/v1/chat/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function chatHealth(): Promise<{ status: string; service: string }> {
  return apiRequest('/api/v1/chat/health', {
    method: 'GET',
  });
}

// ================= PAPERS =================

export async function ingestDocuments(): Promise<IngestionResponse> {
  return apiRequest('/api/v1/papers/ingest', {
    method: 'POST',
  });
}

export async function uploadPDF(file: File): Promise<IngestionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const token = localStorage.getItem("token");

  const response = await fetch(`${API_BASE_URL}/api/v1/papers/upload`, {
    method: 'POST',
    body: formData,
    headers: {
      ...(token && { Authorization: `Bearer ${token}` }),
    },
  });

  if (response.status === 401) {
    logout();
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      detail: response.statusText,
    }));
    throw new Error(errorData.detail || "Upload failed");
  }

  return response.json();
}

export async function getUploadedFiles() {
  const res = await apiRequest<{ status: string; files: any[] }>(
    '/api/v1/papers/files'
  );
  return res.files;
}

export async function deleteUploadedFile(filename: string) {
  const encoded = encodeURIComponent(filename);
  return apiRequest(`/api/v1/papers/files/${encoded}`, {
    method: 'DELETE',
  });
}

export async function searchDocuments(query: string, top_k: number = 5) {
  return apiRequest(
    `/api/v1/papers/search?query=${encodeURIComponent(query)}&top_k=${top_k}`
  );
}

export async function getCollectionStats(): Promise<CollectionStats> {
  return apiRequest('/api/v1/papers/stats');
}

// ================= WORKSPACES =================

export async function getWorkspaces() {
  return apiRequest('/api/v1/workspaces');
}

export async function createWorkspace(name: string) {
  return apiRequest('/api/v1/workspaces', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

// ================= DOCUMENTS =================

export async function getDocuments(workspaceId: string) {
  return apiRequest(`/api/v1/documents?workspace_id=${workspaceId}`);
}

export async function createDocument(workspaceId: string, title: string) {
  return apiRequest('/api/v1/documents', {
    method: 'POST',
    body: JSON.stringify({ workspace_id: workspaceId, title }),
  });
}

export async function deleteDocument(docId: number) {
  return apiRequest(`/api/v1/documents/${docId}`, {
    method: 'DELETE',
  });
}

export async function getDocumentById(docId: number) {
  return apiRequest(`/api/v1/documents/${docId}`);
}

export async function updateDocument(
  docId: number,
  content: string,
  title?: string
) {
  return apiRequest(`/api/v1/documents/${docId}`, {
    method: "PUT",
    body: JSON.stringify({
      content,
      ...(title ? { title } : {}),
    }),
  });
}

// ================= EXPORTS (PDF / DOCX)

export async function exportDocumentPDF(docId: number) {
  const token = localStorage.getItem('token')
  const url = `${API_BASE_URL}/api/export/doc/${docId}/pdf`
  const res = await fetch(url, {
    method: 'GET',
    headers: {
      ...(token && { Authorization: `Bearer ${token}` }),
    },
  })
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText)
    throw new Error(err || 'Export PDF failed')
  }
  return res.blob()
}

export async function exportDocumentDOCX(docId: number) {
  const token = localStorage.getItem('token')
  const url = `${API_BASE_URL}/api/export/doc/${docId}/docx`
  const res = await fetch(url, {
    method: 'GET',
    headers: {
      ...(token && { Authorization: `Bearer ${token}` }),
    },
  })
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText)
    throw new Error(err || 'Export DOCX failed')
  }
  return res.blob()
}

// ================= AI SUGGESTIONS

export async function aiSuggest(content: string, prompt?: string) {
  const q = prompt ? `${prompt}\n\n${content}` : content
  return apiRequest('/api/v1/chat/chat', {
    method: 'POST',
    body: JSON.stringify({ query: q, use_context: false }),
  })
}

