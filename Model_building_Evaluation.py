import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD

RANDOM_STATE = 42
NUM_USERS = 800
MAX_RATING_REPEAT = 20
TOP_N = 5

np.random.seed(RANDOM_STATE)

# ==============================
# 1 Load Dataset
# ==============================

df = pd.read_csv("Book_Features_Phase1.csv")
df = df[['ISBN','Book-Title','avg_rating','rating_count']]
df = df.dropna()

# ==============================
# 2 Simulate User Ratings
# ==============================

df_expanded = df.loc[df.index.repeat(df['rating_count'].clip(upper=MAX_RATING_REPEAT))].copy()
df_expanded['User_ID'] = np.random.randint(1, NUM_USERS, size=len(df_expanded))
df_expanded['User_Rating'] = np.clip(
    df_expanded['avg_rating'] + np.random.normal(loc=0, scale=0.8, size=len(df_expanded)),
    1,
    10
).round(1)

ratings = df_expanded[['User_ID','ISBN','User_Rating']]

# ==============================
# 3 User Item Matrix
# ==============================

user_item = ratings.pivot_table(index='User_ID',
                                columns='ISBN',
                                values='User_Rating')

user_item = user_item.fillna(0)

matrix = user_item.values

# ==============================
# 4 Train Test Mask Split
# ==============================

train = matrix.copy()
test = np.zeros(matrix.shape)

for user in range(matrix.shape[0]):

    ratings_index = matrix[user].nonzero()[0]

    if len(ratings_index) > 1:

        test_ratings = np.random.choice(
            ratings_index,
            size=max(1, int(len(ratings_index)*0.2)),
            replace=False
        )

        train[user, test_ratings] = 0
        test[user, test_ratings] = matrix[user, test_ratings]

# ==============================
# 5 ITEM BASED CF
# ==============================

item_similarity = cosine_similarity(train.T)

denominator = np.abs(item_similarity).sum(axis=1)
denominator[denominator==0] = 1

item_pred = np.dot(train, item_similarity) / denominator
item_pred = np.clip(item_pred, 1, 10)

# ==============================
# 6 USER BASED CF
# ==============================

user_similarity = cosine_similarity(train)

rating_counts = (train != 0).sum(axis=1)
user_means = np.divide(
    train.sum(axis=1),
    rating_counts,
    out=np.zeros(train.shape[0]),
    where=rating_counts != 0
)

ratings_diff = train - user_means[:, None]
ratings_diff[train == 0] = 0

denominator = np.abs(user_similarity).sum(axis=1)
denominator[denominator==0] = 1

user_pred = user_means[:, None] + np.dot(user_similarity, ratings_diff) / denominator[:,None]
user_pred = np.clip(user_pred, 1, 10)

# ==============================
# 7 SVD MATRIX FACTORIZATION
# ==============================

n_components = max(1, min(20, train.shape[0] - 1, train.shape[1] - 1))
svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)

train_centered = train - user_means[:, None]
train_centered[train == 0] = 0

U = svd.fit_transform(train_centered)
Sigma = svd.singular_values_
VT = svd.components_

svd_pred = user_means[:, None] + np.dot(np.dot(U,np.diag(Sigma)),VT)
svd_pred = np.clip(svd_pred, 1, 10)

# ==============================
# 8 Evaluation
# ==============================

def evaluate(pred, test):

    test_indices = test.nonzero()

    if len(test_indices[0]) == 0:
        return np.nan, np.nan

    pred = pred[test_indices].flatten()
    actual = test[test_indices].flatten()

    mse = mean_squared_error(actual,pred)
    mae = mean_absolute_error(actual,pred)

    return mse, mae


item_mse,item_mae = evaluate(item_pred,test)
user_mse,user_mae = evaluate(user_pred,test)
svd_mse,svd_mae = evaluate(svd_pred,test)

print("\nMODEL PERFORMANCE")
print("-----------------------")

print("Item CF")
print("MSE:",item_mse,"MAE:",item_mae)

print("\nUser CF")
print("MSE:",user_mse,"MAE:",user_mae)

print("\nSVD")
print("MSE:",svd_mse,"MAE:",svd_mae)

# ==============================
# 9 Precision@K / Recall@K
# ==============================

def precision_recall_at_k(pred, train, test, k=5, threshold=7.0):

    precisions=[]
    recalls=[]

    for user in range(pred.shape[0]):

        pred_ratings = pred[user].copy()
        true_ratings = test[user]

        pred_ratings[train[user] > 0] = -np.inf

        relevant = np.where(true_ratings>=threshold)[0]

        top_k = np.argsort(pred_ratings)[::-1][:k]

        relevant_recommended = len(set(top_k) & set(relevant))

        precision = relevant_recommended/k

        recall = 0 if len(relevant)==0 else relevant_recommended/len(relevant)

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions),np.mean(recalls)


p_item,r_item = precision_recall_at_k(item_pred, train, test)
p_user,r_user = precision_recall_at_k(user_pred, train, test)
p_svd,r_svd = precision_recall_at_k(svd_pred, train, test)

print("\nPRECISION@5 / RECALL@5")
print("-----------------------")

print("Item CF:",p_item,r_item)
print("User CF:",p_user,r_user)
print("SVD:",p_svd,r_svd)

# ==============================
# 10 Personalized Book Recommendations
# ==============================

def get_recommendations_for_user(user_id, pred_df, user_item_df, books_df, top_n=5):

    if user_id not in pred_df.index:
        raise ValueError(f"User {user_id} not found in prediction matrix.")

    user_scores = pred_df.loc[user_id].copy()
    rated_books = user_item_df.loc[user_id] > 0
    user_scores = user_scores[~rated_books]

    recommended_isbn = user_scores.sort_values(ascending=False).head(top_n).index

    recommendations = books_df[books_df['ISBN'].isin(recommended_isbn)][
        ['Book-Title', 'ISBN', 'avg_rating']
    ].drop_duplicates().copy()

    recommendations['Predicted_Score'] = recommendations['ISBN'].map(user_scores.to_dict())
    recommendations = recommendations.sort_values('Predicted_Score', ascending=False)

    return recommendations.reset_index(drop=True)


item_pred_df = pd.DataFrame(item_pred,
                            index=user_item.index,
                            columns=user_item.columns)

user_pred_df = pd.DataFrame(user_pred,
                            index=user_item.index,
                            columns=user_item.columns)

svd_pred_df = pd.DataFrame(svd_pred,
                           index=user_item.index,
                           columns=user_item.columns)

sample_users = user_item.index[:3]
prediction_frames = {
    'Item CF': item_pred_df,
    'User CF': user_pred_df,
    'SVD': svd_pred_df
}

print("\nPERSONALIZED TOP 5 RECOMMENDATIONS")
print("-----------------------")

for user_id in sample_users:
    print(f"\nUser {user_id}")
    print("=" * 25)

    for model_name, pred_df in prediction_frames.items():
        recommendations = get_recommendations_for_user(
            user_id=user_id,
            pred_df=pred_df,
            user_item_df=user_item,
            books_df=df,
            top_n=TOP_N
        )

        print(f"\n{model_name} Recommendations")
        print(recommendations[['Book-Title', 'ISBN', 'Predicted_Score']].to_string(index=False))