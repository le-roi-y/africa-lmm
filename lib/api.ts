const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Citation {
  document_id: string;
  page_number: number | null;
  chunk_id: string | null;
}

export interface QueryResponse {
  question: string;
  answer: string;
  citations: Citation[];
  retrieved_chunks: unknown[];
}

export interface IngestResponse {
  document_id: string;
  filename: string;
  pages: number;
  message: string;
}

export async function askQuestion(
  question: string,
): Promise<QueryResponse> {
  const response = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Failed to query AFRICA-LMM.");
  }

  return response.json();
}

export async function uploadDocument(
  file: File,
): Promise<IngestResponse> {
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
