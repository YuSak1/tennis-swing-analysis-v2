import axios from "axios";
import type { AnalysisResponse, Hand } from "../types";

const api = axios.create({
  baseURL: "/api",
});

export async function analyzeSwing(
  videoFile: File,
  hand: Hand
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("hand", hand);

  const response = await api.post<AnalysisResponse>("/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000, // 5 min — video processing can be slow
  });
  return response.data;
}

export async function healthCheck(): Promise<{ status: string; version: string }> {
  const response = await api.get<{ status: string; version: string }>("/health");
  return response.data;
}
