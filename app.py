import streamlit as st
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .movie-card {
        background: #1c1f26;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        height: 100%;
    }
    .movie-title {
        font-size: 13px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 8px;
    }
    .movie-rating {
        font-size: 12px;
        color: #f5c518;
    }
    .similarity {
        font-size: 11px;
        color: #aaaaaa;
    }
    h1 { color: #e50914; }
</style>
""", unsafe_allow_html=True)

# ── Load pickles (cached) ──────────────────────────────────────────
@st.cache_resource
def load_data():
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    return indices, tfidf_matrix, df

indices, tfidf_matrix, df = load_data()

# ── TMDB ───────────────────────────────────────────────────────────
TMDB_API_KEY = "a8251c2417ea2f57a1bcc61b273d6064"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://via.placeholder.com/200x300?text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_poster(title: str) -> str:
    try:
        res = requests.get(
            f"{TMDB_BASE}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=5,
        )
        results = res.json().get("results", [])
        if results and results[0].get("poster_path"):
            return TMDB_IMG + results[0]["poster_path"]
    except Exception:
        pass
    return PLACEHOLDER

# ── Recommendation logic ───────────────────────────────────────────
def recommend(title: str, n: int = 10):
    title_lower = title.lower().strip()
    if title_lower not in indices:
        return []
    idx = indices[title_lower]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = np.argsort(sim_scores)[::-1][:n]
    results = []
    for i in top_indices:
        row = df.iloc[i]
        results.append({
            "title": row["title"],
            "score": round(float(sim_scores[i]), 3),
            "rating": row.get("vote_average", 0),
            "genres": row.get("genres", ""),
        })
    return results

# ── UI ─────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown("Find movies similar to what you love — powered by **TF-IDF + Cosine Similarity**")

st.divider()

# Search box with autocomplete from dataset
all_titles = sorted([t.title() for t in indices.keys()])
selected = st.selectbox(
    "🔍 Search for a movie",
    options=[""] + all_titles,
    index=0,
    placeholder="Type a movie name...",
)

n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("🎯 Get Recommendations", use_container_width=True) and selected:
    with st.spinner("Finding similar movies..."):
        recs = recommend(selected, n_recs)

    if not recs:
        st.error(f"Movie **{selected}** not found in the dataset.")
    else:
        # Show selected movie info
        st.subheader(f"Movies similar to **{selected}**")

        with st.expander("📽️ Selected Movie Details", expanded=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                poster = fetch_poster(selected)
                st.image(poster, width=150)
            with col2:
                idx = indices[selected.lower()]
                row = df.iloc[idx]
                st.markdown(f"**Genres:** {row.get('genres', 'N/A')}")
                st.markdown(f"**Rating:** ⭐ {row.get('vote_average', 'N/A')}")
                st.markdown(f"**Overview:** {row.get('overview', 'N/A')[:300]}...")

        st.divider()
        st.subheader("🍿 Recommended Movies")

        # Display in grid of 5 per row
        cols_per_row = 5
        for i in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(recs):
                    movie = recs[i + j]
                    with col:
                        poster = fetch_poster(movie["title"])
                        st.image(poster, use_column_width=True)
                        st.markdown(f"**{movie['title'].title()}**")
                        st.markdown(f"⭐ {movie['rating']} | 🎯 {movie['score']}")
                        st.caption(f"{movie['genres'][:40] if movie['genres'] else ''}")

elif not selected and st.button("🎯 Get Recommendations", use_container_width=True):
    st.warning("Please select a movie first.")

# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built with ❤️ using Streamlit · Data from TMDB · TF-IDF Content-Based Filtering</small></center>",
    unsafe_allow_html=True,
)
