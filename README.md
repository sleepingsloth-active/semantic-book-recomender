
---

## **2️⃣ AI-Powered Book Recommender README**

**Filename:** `README.md`

```markdown
# AI-Powered Book Recommendation System

A semantic search engine and interactive web app to recommend books based on **user queries**. The system uses **vector embeddings** and similarity search to find books matching your interests.

---

## Features
- Preprocesses and cleans book metadata, including descriptions, titles, and ISBNs.
- Generates vector embeddings using **HuggingFace all-MiniLM-L6-v2**.
- Stores embeddings in a **Chroma vector database** for fast semantic search.
- Interactive **Gradio web app** for querying book recommendations.
- Returns book thumbnails, title, author, and short description.

---

## Technologies Used
- Python 3
- Pandas, NumPy
- LangChain
- HuggingFace Transformers
- ChromaDB
- Gradio (for frontend)
- Regex (for data parsing)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/book-recommender.git

##Install dependencies:
pip install pandas numpy gradio langchain chromadb sentence-transformers python-dotenv

##screenshots:
![image alt](https://github.com/sleepingsloth-active/book-recomender/blob/main/Screenshot%20(94).png?raw=true)
![image alt]()

