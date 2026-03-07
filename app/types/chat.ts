export interface ChatSession {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface DBMessage {
  id: number;
  session_id: number;
  role: "user" | "assistant" | "system" | "context";
  content: string;
  is_transcript: boolean;
  created_at: string;
}

export interface UploadedFile {
  id: number;
  session_id: number;
  filename: string;
  category: "audio" | "image" | "text" | "pdf" | "unknown";
  status: "success" | "error";
  chunks: number;
  detail: string | null;
  file_size: number | null;
  created_at: string;
}
