import { queryChromaDB } from "./embeddingService.js";

export async function queryWithQA(query: string, userId: string): Promise<{
  id: string;
  title: string;
  description: string;
  type: string;
  link: string;
} | null> {
  try {
    const bestMatch = await queryChromaDB(query, userId);
    return bestMatch;
  } catch (error) {
    console.error("[ERROR] Error in queryWithQA:", error);
    throw error;
  }
}