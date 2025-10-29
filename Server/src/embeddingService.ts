import { pipeline } from "@xenova/transformers";
import { ChromaClient, IncludeEnum } from "chromadb";
import dotenv from "dotenv";

dotenv.config();

let embedder: any;
let embedderPromise: Promise<any> | null = null;

// Initialize embedder with a promise we can await
async function getEmbedder() {
  if (embedder) return embedder;
  
  if (!embedderPromise) {
    embedderPromise = (async () => {
      try {
        console.log("[INFO] Loading embedding model...");
        embedder = await pipeline('feature-extraction', process.env.EMBEDDING_MODEL, {
          quantized: false,
        });
        console.log("[INFO] Embedding model loaded successfully");
        return embedder;
      } catch (error) {
        console.error("[ERROR] Failed to load embedder:", error);
        embedderPromise = null; // Reset so it can retry
        throw error;
      }
    })();
  }
  
  return embedderPromise;
}

const client = new ChromaClient({
  path: process.env.CHROMA_DB_URL,
});

async function getEmbeddings(text: string): Promise<number[]> {
  try {
    const model = await getEmbedder(); // Wait for model to load
    const embedding = await model(text, { pooling: 'mean', normalize: true });
    const result: number[] = Array.from(embedding.data as Float32Array);
    return result;
  } catch (error) {
    console.error("[ERROR] Error getting embeddings:", error);
    throw error;
  }
}

export async function storeCardEmbeddings(card: {
  _id: string;
  title: string;
  description?: string;
  type: string;
  link?: string;
  userId?: string;
}) {
  try {
    if (!process.env.CHROMA_DB_URL) {
      console.warn("[WARN] ChromaDB URL not configured, skipping embedding storage");
      return;
    }

    const combinedText = `${card.title} ${card.description || ""} ${card.type} ${card.link || ""}`.trim();
    const embedding = await getEmbeddings(combinedText);
    
    const collection = await client.getOrCreateCollection({ 
      name: "content_collection",
      metadata: { "hnsw:space": "cosine" }
    });

    await collection.upsert({
      ids: [card._id],
      embeddings: [embedding],
      documents: [combinedText],
      metadatas: [{
        title: card.title,
        description: card.description || "",
        type: card.type,
        link: card.link || "",
        userId: card.userId || "",
      }],
    });
  } catch (error) {
    console.error("[ERROR] Error storing embeddings:", error);
  }
}

export async function queryChromaDB(query: string, userId: string): Promise<{
  id: string;
  title: string;
  description: string;
  type: string;
  link: string;
} | null> {
  try {
    if (!process.env.CHROMA_DB_URL) {
      console.warn("[WARN] ChromaDB URL not configured, skipping vector search");
      return null;
    }

    const queryEmbedding = await getEmbeddings(query);
    
    let collection;
    try {
      collection = await client.getOrCreateCollection({ 
        name: "content_collection",
        metadata: { "hnsw:space": "cosine" }
      });
    } catch (collectionError: any) {
      console.error("[ERROR] Failed to get/create collection:", collectionError);
      try {
        await client.deleteCollection({ name: "content_collection" });
        collection = await client.createCollection({ 
          name: "content_collection",
          metadata: { "hnsw:space": "cosine" }
        });
      } catch (recreateError) {
        console.error("[ERROR] Failed to recreate collection:", recreateError);
        return null;
      }
    }

    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 1,
      where: { userId },
      include: [IncludeEnum.Metadatas, IncludeEnum.Distances],
    });

    if (!results.ids?.[0]?.[0] || !results.metadatas?.[0]?.[0]) {
      return null;
    }

    const bestMatch = {
      id: results.ids[0][0],
      title: results.metadatas[0][0].title as string,
      description: results.metadatas[0][0].description as string,
      type: results.metadatas[0][0].type as string,
      link: results.metadatas[0][0].link as string,
    };
    return bestMatch;
  } catch (error) {
    console.error("[ERROR] Error querying ChromaDB:", error);
    return null;
  }
}