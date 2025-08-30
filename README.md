
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
## Screenshots
<img width="1486" height="972" alt="Screenshot (94)" src="https://github.com/user-attachments/assets/2877697d-5fa7-4af9-8a2e-530a1af68e69" />
<img width="1500" height="956" alt="Screenshot (95)" src="https://github.com/user-attachments/assets/f6e40391-e4f8-4bc6-9477-54572cdaccd7" />
##Install dependencies:
pip install pandas numpy gradio langchain chromadb sentence-transformers python-dotenv


