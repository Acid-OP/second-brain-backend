import { NextFunction, Request, Response } from "express";
import jwt, { JwtPayload } from "jsonwebtoken";
import dotenv from "dotenv";

dotenv.config();

interface AuthRequest extends Request {
  userId?: string;
}

export const userMiddleware = (req: AuthRequest, res: Response, next: NextFunction) => {
  const header = req.headers.authorization;

  if (!header) {
    res.status(401).json({ message: "No token provided" });
    return;
  }

  const token = header.startsWith("Bearer ") ? header.split(" ")[1] : header;

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as JwtPayload;
    if (!decoded.id) {
      res.status(401).json({ message: "Invalid token payload" });
      return;
    }
    req.userId = decoded.id;
    next();
  } catch (error) {
    console.error("[DEBUG] JWT verification failed:", error);
    res.status(403).json({ message: "Invalid or expired token" });
    return;
  }
};