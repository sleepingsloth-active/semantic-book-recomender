import gradio as gr
import pandas as pd
from dotenv import load_dotenv
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables (for any API keys or sensitive data)
load_dotenv()

# ----------------- DATA LOADING SECTION -----------------
# Load cleaned book data with thumbnails
books = pd.read_csv('books_cleaned.csv')  # Contains book metadata including thumbnails

# ----------------- VECTORSTORE SETUP -----------------
# Initialize embedding model (must match what was used to create the vectorstore)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load persisted Chroma vectorstore (pre-computed embeddings)
db_books = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)


# ----------------- RECOMMENDATION LOGIC -----------------
def get_all_recommendations(docs, books_df, top_k):
    """
    Process similarity search results and extract unique book recommendations.
    Args:
        docs: List of documents from similarity search
        books_df: DataFrame containing book metadata
        top_k: Number of recommendations to return
    Returns:
        DataFrame of recommended books
    """
    seen_isbns = set()  # Track seen ISBNs to avoid duplicates
    recs = []

    for doc in docs:
        try:
            # Extract 13-digit ISBN using regex
            match = re.search(r'\b(\d{13})\b', doc.page_content)
            if match:
                isbn = int(match.group(1))

                if isbn not in seen_isbns:
                    book_row = books_df[books_df["isbn13"] == isbn]
                    if not book_row.empty:
                        recs.append(book_row.iloc[0])
                        seen_isbns.add(isbn)

                if len(recs) >= top_k:
                    break
        except Exception as e:
            print(f"[ERROR] Failed to parse doc: {e}")
            continue

    return pd.DataFrame(recs) if recs else pd.DataFrame()


def recommend_books(query, top_k):
    """
    Main recommendation function that handles:
    - Multiple search attempts if initial results are insufficient
    - Processing and formatting final recommendations
    Args:
        query: User's search query
        top_k: Number of recommendations requested
    Returns:
        HTML formatted results or error message
    """
    multiplier = 2  # Initial multiplier for search space
    max_attempts = 5  # Maximum attempts to find enough unique books
    final_books = pd.DataFrame()

    for attempt in range(max_attempts):
        k_docs = top_k * multiplier  # Expand search space progressively
        docs = db_books.similarity_search(query, k=k_docs)
        print(f"[DEBUG] Attempt {attempt + 1}: Retrieved {len(docs)} docs with k={k_docs}")

        final_books = get_all_recommendations(docs, books, top_k)
        print(f"[DEBUG] Got {len(final_books)} unique recommendations")

        if len(final_books) >= top_k:
            break
        multiplier += 1

    if final_books.empty:
        return "No recommendations found. Try a different query."

    # Format results as HTML cards
    results = []
    for _, row in final_books.iterrows():
        image_url = row.get('thumbnail', None)
        title = row.get('title', 'Unknown Title')
        author = row.get('authors', 'Unknown Author')
        description = str(row.get('description', 'No description available'))[:150] + "..."

        if image_url and isinstance(image_url, str) and image_url.startswith("http"):
            html_block = f"""
            <div style="text-align:center; max-width:200px; margin: 10px; padding: 10px; 
                        background: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <img src="{image_url}" alt="{title} cover" 
                     style="width:150px; height:auto; border-radius:8px; margin-bottom: 10px;">
                <p style="margin: 5px 0;"><b>{title}</b></p>
                <p style="margin: 5px 0; color: #555;"><i>{author}</i></p>
                <p style="margin: 5px 0; font-size: 0.9em; color: #666;">{description}</p>
            </div>
            """
            results.append(html_block)

    return "<div style='display:flex; flex-wrap: wrap; justify-content: center; gap: 20px;'>" + \
        "".join(results) + "</div>"


# ----------------- UI ENHANCEMENTS -----------------
custom_css = """
/* Main container styling */
.gradio-container {
    font-family: 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa;
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

/* Header section */
.header {
    text-align: center;
    margin-bottom: 30px;
}
.header h1 {
    color: #2c3e50;
    font-size: 2.5rem;
    margin-bottom: 10px;
}
.header p {
    color: #7f8c8d;
    font-size: 1.1rem;
}

/* Input section styling */
.input-section {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 30px;
}

/* Search box styling */
.search-box textarea {
    border-radius: 8px !important;
    padding: 15px !important;
    border: 1px solid #ddd !important;
    font-size: 1rem !important;
    min-height: 100px !important;
}
.search-box textarea:focus {
    border-color: #3498db !important;
    box-shadow: 0 0 0 2px rgba(52,152,219,0.2) !important;
}

/* Slider customization */
.slider-container .wrap {
    margin-top: 20px;
}
.slider-container input[type=range] {
    height: 8px;
}
.slider-container .track {
    background: #bdc3c7;
}
.slider-container .track-fill {
    background: #3498db;
}
.slider-container .thumb {
    width: 20px;
    height: 20px;
    background: #3498db;
    border: 3px solid white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Results section */
.results {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
"""

# ----------------- GRADIO INTERFACE -----------------
with gr.Blocks(css=custom_css, title="Book Recommendation Engine") as demo:
    # Header Section
    with gr.Column(elem_classes=["header"]):
        gr.Markdown("# 📚 Semantic Book Recommender")
        gr.Markdown("Discover your next favorite book with AI-powered recommendations")

    # Input Section
    with gr.Column(elem_classes=["input-section"]):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    lines=3,
                    placeholder="Describe a book you'd like (genre, theme, writing style)...",
                    label="What are you in the mood to read?",
                    elem_classes=["search-box"]
                )
            with gr.Column():
                slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Recommendations",
                    interactive=True
                )
                gr.Markdown("<small>Adjust the slider to get more or fewer recommendations</small>")

        submit_btn = gr.Button("Find Recommendations", variant="primary")

    # Results Section
    with gr.Column(elem_classes=["results"]):
        gr.Markdown("## Recommended Books")
        output = gr.HTML()

    # Event Handling
    submit_btn.click(
        fn=recommend_books,
        inputs=[query_input, slider],
        outputs=output
    )
    query_input.submit(
        fn=recommend_books,
        inputs=[query_input, slider],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()