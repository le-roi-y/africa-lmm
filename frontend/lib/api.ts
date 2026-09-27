const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Citation {
  document_id: string;
  page_number: number | null;
  chunk_id: string | null;
}

export interface RetrievedChunk {
  chunk: {
    chunk_id?: string;
    document_id?: string;
    text?: string;
    page_number?: number | null;
    metadata?: Record<string, unknown>;
  };
  score: number;
  rank: number;
}

export interface QueryResponse {
  question: string;
  answer: string;
  conversation_id: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
}

export interface QueryOptions {
  conversationId?: string;
  documentIds?: string[];
}

export interface IngestResponse {
  document_id: string;
  filename: string;
  pages: number;
  message: string;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  pages: number;
  size_bytes: number;
  indexed: boolean;
}

export interface DocumentsResponse {
  documents: DocumentInfo[];
  total: number;
}

export async function askQuestion(
  question: string,
  options: QueryOptions = {},
): Promise<QueryResponse> {
  const response = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: 5,
      conversation_id: options.conversationId ?? null,
      document_ids: options.documentIds ?? [],
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to query AFRICA-LMM.");
  }

  return response.json();
}

export async function uploadDocument(file: File): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to upload document.");
  }

  return response.json();
}

export async function getDocuments(): Promise<DocumentsResponse> {
  const response = await fetch(`${API_URL}/documents`, {
    cache: "no-store",
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to load documents.");
  }

  return response.json();
}

export function getDocumentUrl(
  documentId: string,
  pageNumber?: number | null,
): string {
  const base =
    `${API_URL}/documents/${encodeURIComponent(documentId)}/file`;

  return pageNumber
    ? `${base}#page=${pageNumber}`
    : base;
}

export async function getHealth(): Promise<{
  status: string;
  service: string;
  version: string;
}> {
  const response = await fetch(`${API_URL}/health`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error("API unavailable.");
  }

  return response.json();
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ConversationMessage[];
}

export interface ConversationsResponse {
  conversations: ConversationSummary[];
  total: number;
}

export async function createConversation(
  title = "Nouvelle conversation",
): Promise<ConversationDetail> {
  const response = await fetch(`${API_URL}/conversations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to create conversation.");
  }

  return response.json();
}

export async function getConversations(): Promise<ConversationsResponse> {
  const response = await fetch(`${API_URL}/conversations`, {
    cache: "no-store",
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to load conversations.");
  }

  return response.json();
}

export async function getConversation(
  conversationId: string,
): Promise<ConversationDetail> {
  const response = await fetch(
    `${API_URL}/conversations/${encodeURIComponent(conversationId)}`,
    {
      cache: "no-store",
    },
  );

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to load conversation.");
  }

  return response.json();
}

export async function deleteConversation(
  conversationId: string,
): Promise<void> {
  const response = await fetch(
    `${API_URL}/conversations/${encodeURIComponent(conversationId)}`,
    {
      method: "DELETE",
    },
  );

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to delete conversation.");
  }
}
