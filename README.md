<div align="center">
  <img src="https://github.com/Acid-OP/second-brain/blob/main/Client/public/brain.png?raw=true" 
       alt="Second Brain Logo" 
       width="40">
</div>

# Second Brain 🧠 - Your AI-Powered Knowledge Vault

> **Never lose an important link again** - Store, organize and instantly retrieve your saved content from Twitter, YouTube, Reddit and more using intelligent embeddings.

---

## ✨ Key Features

Second Brain is an intelligent knowledge management platform that helps you store, organize, and retrieve important content from various platforms using AI-powered embeddings. Think of it as your external memory bank for digital content.
---
## 📂 Project Repositories

This project is split into two repositories for better separation of concerns:

### 🖥️ Frontend (Client)
```bash
https://github.com/Acid-OP/second-brain-frontend
```
###⚙️ Backend (Server)
```bash
https://github.com/Acid-OP/second-brain-backend
```
## 🌟 Key Features

### 🗄️ Smart Content Storage
- Save important links from multiple platforms:
  - **Twitter/X** posts
  - **YouTube** videos
  - **Reddit** threads
  - **General web links**
- Store content with rich metadata:
  - Custom titles
  - Descriptions
  - Content type classification
  - AI-generated embeddings for powerful search

### 🔍 Intelligent Search
- Find your saved content effortlessly:
  - Search by title, description, or content characteristics
  - Uses open-source embedding models (not commercial LLMs)
  - Note: While highly effective, results may vary occasionally due to the nature of open-source models
    
### 💠 Vector Search Powered by ChromaDB
- Lightning-fast semantic search using ChromaDB vector database
- Persistent storage for your embeddings
- Automatic index management
  
### 🧠 Share Your Brain
- Create shareable versions of your knowledge repository:
  - Generate public links to showcase your collections
  - Toggle privacy settings to make collections private
  - Control exactly what others can see

### 🛠️ Content Management
- Full CRUD functionality:
  - Create new knowledge cards
  - Share your cards.
  - Delete content you no longer need
  - Organize your knowledge base

### 🔒 Secure Access
- Complete user authentication:
  - Secure signup/login
  - Protected content storage
  - Logout functionality

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v16 or higher)
- **npm** or **yarn** package manager
- **MongoDB Atlas** account (for database storage)
- **Embedding API Key** (choose one of these options):
  - Preferred: Any LLM API key (OpenAI, Hugging Face, etc.)
  - Alternative: We've pre-configured support for the open-source `sentence-transformers/all-MiniLM-L6-v2` model
- **ChromaDB** (either local or cloud instance) 
  - Local: `pip install chromadb` (Python 3.7+ required)
  - Cloud: Free tier available at [Chroma Cloud](https://www.trychroma.com/)

#### About Our Model Choice:
We default to using the open-source `all-MiniLM-L6-v2` sentence transformer because:
- 🆓 No API costs or usage limits
- 🔒 Runs completely locally (no data sent externally)
- ⚖️ Balanced 384-dimensional embeddings
- 📏 Good performance for semantic search tasks

For better results, you can optionally configure:
1. OpenAI embeddings (requires API key)
2. Hugging Face inference API
3. Any other compatible embedding service

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/second-brain.git
   cd second-brain
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### 3. Environment Variables Setup

Create these `.env` files in their respective directories:

#### For Local Development:
```env
# Frontend (create in /Client/.env)
VITE_BACKEND_URL=http://localhost:5000 # or Vercel deployed Backend URL

# Backend (create in /Server/.env)
MONGO_URI=your_mongodb_atlas_connection_string
JWT_SECRET=your_secure_random_string
PORT=YOUR PORT
FRONTEND_URL=http://localhost:3000 # or Vercel deployed Frontend URL
EMBEDDING_MODEL=all-MiniLM-L6-v2  # or 'openai' for OpenAI
# OPENAI_API_KEY=sk-your-key-here  # Uncomment if using OpenAI
# ChromaDB Configuration 
CHROMA_DB_URL=http://localhost:8000  # For local ChromaDB
# OR for Chroma Cloud:
# CHROMA_DB_URL=https://your-cluster-id.chroma.cloud

4. Run the development server:
   ```bash
   npm run dev
   ```

## 🧩 How It Works

### Saving Content
1. Navigate to the "Add Content" section
2. Enter:
   - Title (what you'll remember it by)
   - URL (the original content link)
   - Description (key points to remember)
   - Content type (YouTube, Twitter, Reddit, or General)
3. The system automatically generates AI embeddings for powerful search

### Finding Content
1. Use the search functionality
2. Enter any details you remember:
   - Exact title or description words
   - Related concepts or topics
   - Content characteristics
3. The system will surface the most relevant matches

### Sharing Collections
1. Go to "Share Brain" section
2. Select a collection to share
3. Click "Share" (By default it is Private):
   - Public: Anyone with link can view
   - Private: Only you can access
4. Copy the shareable link to distribute
5. Click on "Make it Private" to stop the Share functionality

## 🛡️ Privacy & Security

- Your data is stored securely in MongoDB Atlas
- Authentication uses JWT tokens
- Passwords are hashed before storage
- You maintain full control over shared content

## 🌐 Technologies Used

### Frontend
- React.js
- TypeScript
- Tailwind CSS
- Framer Motion (animations)
- Axios (API calls)
  

### Backend
- Node.js
- TypeScript
- Express.js
- Auth(JWT, ZOD)
- MongoDB (database)
- Mongoose (ODM)

### AI Components
- Open source model for embeddings generation. 
- Custom similarity search algorithms

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

For questions or support, contact us at:
- Email: gauravkapur596@gmail.com
- Twitter: https://x.com/GauravKapurr

---

**Second Brain** - Never lose an important idea or resource again. Your digital memory, enhanced with AI.
