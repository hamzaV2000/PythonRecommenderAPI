import warnings

import pandas as pd
from flask_restful import Api, Resource
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, Response
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


class RecommendBySimilarBooks(Resource):
    def get(self, title, domain=0):
        return get_recommendations(title, domain)


class MostRated(Resource):
    def get(self, n):
        return get_topN(n)


class RecommendByAuthor(Resource):
    def get(self, author, n):
        return get_recommendationsByAuthor(author, n)


def get_recommendations(title="empty", domain=0):
    # from sklearn.metrics.pairwise import  linear_kernel
    df = pd.read_csv('books.csv').head((domain + 1) * 500)
    # Replace NaN with an empty string
    df['description'] = df['description'].fillna('')

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    print(title)
    idx = ""
    try:
        idx = indices[title]
    except:
        if (domain + 1) * 500 < 52000:
            return get_recommendations(title, domain + 1)
        else:
            return "not found"

    # Create a Tfidf Vectorizer and Remove stopwords
    tfidf = TfidfVectorizer(stop_words='english')
    # Fit and transform the data to a tfidf matrix
    tfidf_matrix = tfidf.fit_transform(df['genres'])

    # Print the shape of the tfidf_matrix
    print(tfidf_matrix.shape)

    # Compute the cosine similarity between each movie description
    # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim2 = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))
    # Sort the movies based on the similarity scores

    try:
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    except ValueError:
        return "not found"

    # Get the scores of the 10 most similar movies
    top_similar = sim_scores[1:10 + 1]
    # Get the movie indices
    movie_indices = [i[0] for i in top_similar]
    # Return the top 10 most similar movies
    print("Returning Recommendations")
    return Response(df.iloc[movie_indices][["title", "genres", "coverImg"]].to_json(orient="records"),
                    mimetype='application/json')


def get_recommendationsByAuthor(author="empty", n=5):
    # from sklearn.metrics.pairwise import  linear_kernel
    df = pd.read_csv('books.csv')
    df = df[df["author"] == author]
    return Response(df.nlargest(n, 'numRatings')[["title", "genres", "coverImg", "rating"]].to_json(orient="records"),
                    mimetype='application/json')


def get_topN(n):
    df = pd.read_csv('books.csv')
    return Response(df.nlargest(n, 'numRatings')[["title", "genres", "coverImg", "rating"]].to_json(orient="records"),
                    mimetype='application/json')


app = Flask(__name__)
api = Api(app)
api.add_resource(RecommendBySimilarBooks, "/recommendBySimilarBooks/<string:title>/<int:domain>")
api.add_resource(RecommendByAuthor, "/recommendByAuthor/<string:author>/<int:n>")
api.add_resource(MostRated, "/mostRated/<int:n>")
if __name__ == "__main__":
    app.run(debug=True)
