import warnings

import pandas as pd
from flask_restful import Api, Resource
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, Response
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# ##############################################################################################################################
dff = pd.read_csv('books.csv')
dff.drop_duplicates(subset=['title'])


def get_recommendationsByBookTitle(title="empty", domain=1):
    print(title, ((domain - 1) * 500), ((domain + 1) * 500))
    # from sklearn.metrics.pairwise import  linear_kernel
    print(domain)
    if dff[dff['title'] == title].shape[0] == 0:
        return "no such title"
    df = pd.DataFrame(dff[(domain - 1) * 500: (domain + 1) * 500])
    df['description'] = df['description'].fillna('')
    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = ""
    try:
        idx = indices[title]
    except:
        if (domain + 1) * 500 < 52500:
            return get_recommendationsByBookTitle(title, domain + 1)
        else:
            return "no more books"
    print("before tokenizing...........................")
    # Create a Tfidf Vectorizer and Remove stopwords
    tfidf = TfidfVectorizer(stop_words='english')
    # Fit and transform the data to a tfidf matrix
    tfidf_matrix = tfidf.fit_transform(df['description'] + df['genres'])
    print("after tokenizing.............................")
    # Print the shape of the tfidf_matrix
    print(tfidf_matrix.shape)
    print("before cosine")
    cosine_sim2 = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("after cosine")

    print(idx)
    sim_scores = list(enumerate(cosine_sim2[idx]))

    print("after enumeration")
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

    return Response(df.iloc[movie_indices].to_json(orient="records"),
                    mimetype='application/json')


def get_recommendationsByAuthor(author="empty", n=5):
    print("Books for " + author)
    df = pd.read_csv('books.csv')
    df = df[df["author"] == author]
    return Response(df.nlargest(n, 'numRatings').to_json(orient="records"),
                    mimetype='application/json')


def get_recommendationsByGenre(genre="empty", n=5):
    print("Books for " + genre)
    df = pd.read_csv('books.csv')
    df = df[df["genres"].str.contains(genre)]

    return Response(df.nlargest(n, 'numRatings').to_json(orient="records"),
                    mimetype='application/json')


def get_topN(n):
    print("Top ", n, " books")
    df = pd.read_csv('books.csv')
    return Response(df.nlargest(n, 'numRatings').to_json(orient="records"),
                    mimetype='application/json')


# #############################################################################################################################
class RecommendBySimilarBooks(Resource):
    def get(self, title, domain):
        return get_recommendationsByBookTitle(title, domain)


class MostRated(Resource):
    def get(self, n):
        return get_topN(n)


class RecommendByAuthor(Resource):
    def get(self, author, n):
        return get_recommendationsByAuthor(author, n)


class RecommendByGenre(Resource):
    def get(self, genre, n):
        return get_recommendationsByGenre(genre, n)


app = Flask(__name__)
api = Api(app)
api.add_resource(RecommendBySimilarBooks, "/recommendBySimilarBooks/<string:title>/<int:domain>")
api.add_resource(RecommendByAuthor, "/recommendByAuthor/<string:author>/<int:n>")
api.add_resource(RecommendByGenre, "/recommendByGenre/<string:genre>/<int:n>")
api.add_resource(MostRated, "/mostRated/<int:n>")

if __name__ == "__main__":
    app.run(debug=True)
