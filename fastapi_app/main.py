from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Movie Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pickle files ──────────────────────────────────────────────
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("df.pkl", "rb") as f:
    df = pickle.load(f)

# ── TMDB config ────────────────────────────────────────────────────
TMDB_API_KEY = "a8251c2417ea2f57a1bcc61b273d6064"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


# ── Helper: fetch poster + details from TMDB ───────────────────────
def fetch_tmdb_data(title: str) -> dict:
    try:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if data.get("results"):
            movie = data["results"][0]
            poster = (
                TMDB_IMAGE_BASE + movie["poster_path"]
                if movie.get("poster_path")
                else None
            )
            return {
                "tmdb_id": movie.get("id"),
                "poster_url": poster,
                "release_date": movie.get("release_date", "N/A"),
                "overview": movie.get("overview", ""),
                "rating": movie.get("vote_average", 0),
            }
    except Exception:
        pass
    return {"tmdb_id": None, "poster_url": None, "release_date": "N/A", "overview": "", "rating": 0}


# ── Core recommendation logic ──────────────────────────────────────
def get_recommendations(title: str, n: int = 10):
    title_lower = title.lower().strip()
    if title_lower not in indices:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in dataset.")

    idx = indices[title_lower]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1  # exclude the movie itself
    top_indices = np.argsort(sim_scores)[::-1][:n]

    recommendations = []
    for i in top_indices:
        rec_title = df.iloc[i]["title"]
        score = float(sim_scores[i])
        tmdb = fetch_tmdb_data(rec_title)
        recommendations.append({
            "title": rec_title,
            "similarity_score": round(score, 4),
            "genres": df.iloc[i].get("genres", ""),
            "vote_average": df.iloc[i].get("vote_average", 0),
            **tmdb,
        })
    return recommendations


# ── Routes ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Movie Recommender API is running!"}


@app.get("/recommend/{title}")
def recommend(title: str, n: int = 10):
    recs = get_recommendations(title, n)
    return {"query": title, "recommendations": recs}


@app.get("/search")
def search(q: str):
    """Search for movie titles in the dataset"""
    q_lower = q.lower()
    matches = [t for t in indices.keys() if q_lower in t][:20]
    return {"results": matches}


@app.get("/movie/{title}")
def movie_details(title: str):
    """Get details of a specific movie"""
    title_lower = title.lower().strip()
    if title_lower not in indices:
        raise HTTPException(status_code=404, detail="Movie not found")
    idx = indices[title_lower]
    row = df.iloc[idx]
    tmdb = fetch_tmdb_data(title)
    return {
        "title": row["title"],
        "overview": row.get("overview", ""),
        "genres": row.get("genres", ""),
        "vote_average": row.get("vote_average", 0),
        "popularity": row.get("popularity", 0),
        **tmdb,
    }
