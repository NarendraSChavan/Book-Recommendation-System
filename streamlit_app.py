import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

RANDOM_STATE = 42
NUM_USERS = 800
MAX_RATING_REPEAT = 20
TOP_N = 5

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #1f3b73 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: #dbeafe;
        font-size: 1rem;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df = pd.read_csv("Book_Features_Phase1.csv")
    return df[["ISBN", "Book-Title", "avg_rating", "rating_count"]].dropna()


@st.cache_data
def build_recommender(df):
    np.random.seed(RANDOM_STATE)

    df_expanded = df.loc[df.index.repeat(df["rating_count"].clip(upper=MAX_RATING_REPEAT))].copy()
    df_expanded["User_ID"] = np.random.randint(1, NUM_USERS, size=len(df_expanded))
    df_expanded["User_Rating"] = np.clip(
        df_expanded["avg_rating"] + np.random.normal(0, 0.8, size=len(df_expanded)), 1, 10
    ).round(1)

    ratings = df_expanded[["User_ID", "ISBN", "User_Rating"]]
    user_item = ratings.pivot_table(index="User_ID", columns="ISBN", values="User_Rating").fillna(0)
    train = user_item.values

    rating_counts = (train != 0).sum(axis=1)
    user_means = np.divide(
        train.sum(axis=1),
        rating_counts,
        out=np.zeros(train.shape[0]),
        where=rating_counts != 0,
    )

    train_centered = train - user_means[:, None]
    train_centered[train == 0] = 0

    n_components = max(1, min(20, train.shape[0] - 1, train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    U = svd.fit_transform(train_centered)
    Sigma = svd.singular_values_
    VT = svd.components_

    svd_pred = user_means[:, None] + np.dot(np.dot(U, np.diag(Sigma)), VT)
    svd_pred = np.clip(svd_pred, 1, 10)

    pred_df = pd.DataFrame(svd_pred, index=user_item.index, columns=user_item.columns)
    return user_item, pred_df


def get_recommendations_for_user(user_id, pred_df, user_item, books_df, top_n=TOP_N):
    user_scores = pred_df.loc[user_id].copy()
    user_scores = user_scores[user_item.loc[user_id] == 0]

    top_scores = user_scores.sort_values(ascending=False).head(top_n)
    recommendations = books_df[books_df["ISBN"].isin(top_scores.index)][
        ["Book-Title", "ISBN", "avg_rating", "rating_count"]
    ]
    recommendations = recommendations.drop_duplicates().copy()
    recommendations["Predicted_Score"] = recommendations["ISBN"].map(top_scores.to_dict())
    return recommendations.sort_values("Predicted_Score", ascending=False).reset_index(drop=True)


with st.spinner("Preparing personalized recommendations..."):
    df = load_data()
    user_item, pred_df = build_recommender(df)

st.markdown(
    """
    <div class="hero">
        <h1>📚 Book Recommendation System</h1>
        <p>Select a user ID and explore personalized book recommendations in a polished dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Recommendation Controls")
    selected_user = st.selectbox("User ID", user_item.index.tolist())
    top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=TOP_N)
    search_text = st.text_input("Search in recommended titles", placeholder="Type a book title...")

recommendations = get_recommendations_for_user(selected_user, pred_df, user_item, df, top_n=top_n)

if search_text:
    recommendations = recommendations[
        recommendations["Book-Title"].str.contains(search_text, case=False, na=False)
    ].reset_index(drop=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader(f"Recommended Books for User {selected_user}")

if recommendations.empty:
    st.info("No recommendations matched your current search. Try a different title or user ID.")
else:
    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Book-Title": st.column_config.TextColumn("Book Title", width="large"),
            "ISBN": st.column_config.TextColumn("ISBN", width="medium"),
            "avg_rating": st.column_config.NumberColumn("Avg Rating", format="%.1f"),
            "rating_count": st.column_config.NumberColumn("Ratings", format="%d"),
            "Predicted_Score": st.column_config.ProgressColumn(
                "Match Score", min_value=0.0, max_value=10.0, format="%.2f"
            ),
        },
    )

    csv_data = recommendations.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recommendations",
        data=csv_data,
        file_name=f"user_{selected_user}_recommendations.csv",
        mime="text/csv",
    )

st.markdown("</div>", unsafe_allow_html=True)