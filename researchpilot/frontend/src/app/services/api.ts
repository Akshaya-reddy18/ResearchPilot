/**
 * API Service for ResearchPilot Backend
 * Handles all HTTP requests to the FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

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

export interface CollectionStats {
  status: string;
  data: {
    collection_name: string;
    document_count: number;
    embedding_model: number;
    db_path: string;
  };
}

/**
 * Make an API request with error handling
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  // Only set Content-Type for JSON requests (not for FormData)
  const defaultHeaders: HeadersInit = {};
  if (!(options.body instanceof FormData)) {
    defaultHeaders['Content-Type'] = 'application/json';
  }

  try {
    console.log('API Request:', { url, method: options.method || 'GET' });
    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    console.log('API Response:', { status: response.status, statusText: response.statusText });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      console.error('API Error:', errorData);
      
      // Extract error message from nested error objects (Groq API format)
      let errorMessage = errorData.detail || response.statusText;
      if (typeof errorMessage === 'string' && errorMessage.includes('error')) {
        try {
          // Try to parse nested error structure
          const parsed = JSON.parse(errorMessage.replace(/^Error code: \d+ - /, ''));
          if (parsed.error && parsed.error.message) {
            errorMessage = parsed.error.message;
          }
        } catch {
          // If parsing fails, use the original message
        }
      }
      
      throw new Error(errorMessage || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('API Success:', data);
    return data;
  } catch (error) {
    console.error('API Request Failed:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Network error occurred');
  }
}

/**
 * Chat API - Send a query to the research assistant
 */
export async function chat(request: ChatRequest): Promise<ChatResponse> {
  return apiRequest<ChatResponse>('/api/v1/chat/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Papers API - Ingest documents from data directory
 */
export async function ingestDocuments(): Promise<IngestionResponse> {
  return apiRequest<IngestionResponse>('/api/v1/papers/ingest', {
    method: 'POST',
  });
}

/**
 * Papers API - Upload a PDF file
 */
export async function uploadPDF(file: File): Promise<IngestionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const url = `${API_BASE_URL}/api/v1/papers/upload`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - browser will set it with boundary for FormData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Network error occurred');
  }
}

/**
 * Papers API - Search for similar documents
 */
export async function searchDocuments(query: string, top_k: number = 5): Promise<SearchResponse> {
  return apiRequest<SearchResponse>(`/api/v1/papers/search?query=${encodeURIComponent(query)}&top_k=${top_k}`, {
    method: 'GET',
  });
}

/**
 * Papers API - Get collection statistics
 */
export async function getCollectionStats(): Promise<CollectionStats> {
  return apiRequest<CollectionStats>('/api/v1/papers/stats', {
    method: 'GET',
  });
}

/**
 * Chat API - Health check
 */
export async function chatHealth(): Promise<{ status: string; service: string }> {
  return apiRequest<{ status: string; service: string }>('/api/v1/chat/health', {
    method: 'GET',
  });
}

