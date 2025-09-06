import { VercelRequest, VercelResponse } from "@vercel/node";
import express from "express";
import Jwt from "jsonwebtoken";
import { UserModel, ContentModel, LinkModel } from "./db.js";
import { userMiddleware } from "./middelware.js";
import { random } from "./utils.js";
import cors from "cors";
import { storeCardEmbeddings } from "./embeddingService.js";
import { queryWithQA } from "./qaService.js";
import { z } from "zod";
import bcrypt from "bcrypt";
import dotenv from "dotenv";
import helmet from "helmet";

dotenv.config();
const app = express();

const allowedOrigins = [
  "https://secondbrain-hazel.vercel.app",
  "https://second-brain-backend-beige.vercel.app",
  "http://localhost:5173"
]

const corsOptions = {
  origin: function (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  },
  credentials: true,
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
  allowedHeaders: ["Content-Type", "Authorization"],
  optionsSuccessStatus: 200 // For legacy browser support
};

app.use(express.json());
app.use(helmet());
app.use(cors(corsOptions));

// Add OPTIONS handler for preflight requests
app.options("*", cors(corsOptions));

// Zod Schemas for validation
const signupSchema = z.object({
  username: z
    .string()
    .min(3, "Username must be at least 3 characters")
    .max(50, "Username must be 50 characters or less")
    .regex(/^[a-zA-Z][a-zA-Z0-9_]*$/, "Username must start with a letter and contain only letters, numbers, or underscores"),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .regex(/[A-Z]/, "Password must contain at least one uppercase letter")
    .regex(/[a-z]/, "Password must contain at least one lowercase letter")
    .regex(/[0-9]/, "Password must contain at least one number")
    .regex(/[!@#$%^&*]/, "Password must contain at least one special character (!@#$%^&*)"),
});

const signinSchema = z.object({
  username: z
    .string()
    .min(3, "Username must be at least 3 characters")
    .regex(/^[a-zA-Z][a-zA-Z0-9_]*$/, "Username must start with a letter and contain only letters, numbers, or underscores"),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .regex(/[A-Z]/, "Password must contain at least one uppercase letter")
    .regex(/[a-z]/, "Password must contain at least one lowercase letter")
    .regex(/[0-9]/, "Password must contain at least one number")
    .regex(/[!@#$%^&*]/, "Password must contain at least one special character (!@#$%^&*)"),
});

// Signup
app.post("/api/v1/signup", async (req, res) => {
  const username = req.body.username;
  const password = req.body.password;

  const validation = signupSchema.safeParse({ username, password });
  if (!validation.success) {
    res.status(400).json({ message: "Invalid input", errors: validation.error.errors });
  }

  try {
    const saltRounds = 10;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    await UserModel.create({
      username: username,
      password: hashedPassword,
    });
    res.json({ message: "user signed up" });
  } catch (e) {
    console.error("[ERROR] Signup failed:", e);
    res.status(409).json({ message: "User already exists" });
  }
});

// Signin
app.post("/api/v1/signin", async (req, res) => {
  const username = req.body.username;
  const password = req.body.password;

  const validation = signinSchema.safeParse({ username, password });
  if (!validation.success) {
    res.status(400).json({ message: "Invalid input", errors: validation.error.errors });
  }

  const existingUser = await UserModel.findOne({ username });
  if (existingUser && existingUser.password && typeof existingUser.password === "string") {
    const isPasswordValid = await bcrypt.compare(password, existingUser.password);
    if (isPasswordValid) {
      const token = Jwt.sign({ id: existingUser._id }, process.env.JWT_SECRET as string , { expiresIn: "7d" });
      res.json({ token });
    } else {
      res.status(403).json({ message: "Incorrect credentials" });
    }
  } else {
    res.status(403).json({ message: "Incorrect credentials" });
  }
});

// Query
app.post("/api/v1/query", userMiddleware, async (req, res) => {
  const { query } = req.body;
  const userId = (req as any).userId;

  try {
    const card = await queryWithQA(query, userId);
    res.json({ card });
  } catch (error) {
    console.error("[ERROR] Error processing query:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Get Content
app.get("/api/v1/content", userMiddleware, async (req, res) => {
  const userId = (req as any).userId;
  const content = await ContentModel.find({ userId: userId }).populate("userId", "username");
  res.json(content);
});

// Add Content
app.post("/api/v1/content", userMiddleware, async (req, res) => {
  const { link, type, description, title } = req.body;
  const userId = (req as any).userId;

  try {
    const content = await ContentModel.create({
      link,
      type,
      description,
      title,
      userId,
      tags: [],
    });

    try {
      await storeCardEmbeddings({
        _id: content._id.toString(),
        title,
        description,
        type,
        link,
        userId,
      });
    } catch (embeddingError) {
      console.warn("[WARN] Embeddings failed but content saved:", embeddingError);
    }

    res.json({ message: "Content added", _id: content._id });
  } catch (e) {
    console.error("[ERROR] Error adding content:", e);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Delete Content
app.delete("/api/v1/content", userMiddleware, async (req, res) => {
  const { id } = req.body;
  const userId = (req as any).userId;

  try {
    const content = await ContentModel.findOne({ _id: id, userId });
    if (!content) {
      res.status(404).json({ error: "Content not found or you don’t have permission." });
    }
    await ContentModel.deleteOne({ _id: id, userId });
    res.status(200).json({ message: "Content deleted successfully", _id: id });
  } catch (e) {
    console.error("[ERROR] Error deleting content:", e);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Share Functionality
app.post("/api/v1/brain/share", userMiddleware, async (req, res) => {
  const { share } = req.body;
  const userId = (req as any).userId;

  try {
    const existingLink = await LinkModel.findOne({ userId });

    if (share === undefined) {
      res.status(400).json({ error: "Missing 'share' parameter" });
    }

    if (share) {
      if (existingLink) {
        res.status(200).json({
          link: `${process.env.FRONTEND_URL}/brain/${existingLink.hash}`,
        });
      }
      const hash = random(10);
      await LinkModel.create({
        hash,
        userId,
      });
      res.status(200).json({
        link: `${process.env.FRONTEND_URL}/brain/${hash}`,
      });
    } else {
      if (existingLink) {
        await LinkModel.deleteOne({ userId });
      }
      res.status(200).json({
        message: "Your content is now private",
      });
    }
  } catch (e) {
    console.error("[ERROR] Share endpoint error:", e);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Get Shared Content
// @ts-ignore
app.get("/api/v1/brain/:shareLink", async (req, res) => {
  const hash = req.params.shareLink;

  try {
    const link = await LinkModel.findOne({ hash });
    if (!link) {
      return res.status(404).json({
        error: "Share link doesn't exist or has expired",
      });
    }
    const content = await ContentModel.find({ userId: link.userId });
    const user = await UserModel.findOne({ _id: link.userId });

    if (!user) {
      return res.status(404).json({
        error: "User not found",
      });
    }
    res.status(200).json({
      username: user.username,
      contents: content,
    });
  } catch (e) {
    console.error("[ERROR] Brain endpoint error:", e);
    res.status(500).json({ error: "Internal server error" });
  }
});


app.listen(process.env.PORT!, () => {
  console.log("Server running on port 3000");
});