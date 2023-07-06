from flask import Flask, Blueprint, render_template, request, jsonify
import json
import findspark
from pyspark import SparkContext, SparkConf
from engine import RecommendationEngine

main = Blueprint('main', __name__)

@main.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@main.route("/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    movie = recommendation_engine.get_movie(movie_id)
    if movie is None:
        return jsonify(error="Film non trouvé"), 404
    else:
        return jsonify(movie)

@main.route("/newratings/<int:user_id>", methods=["POST"])
def new_ratings(user_id):
    if not recommendation_engine.is_user_known(user_id):
        user_id = recommendation_engine.create_user(user_id)
    ratings = request.json
    recommendation_engine.add_ratings(user_id, ratings)
    return str(user_id)

@main.route("/<int:user_id>/ratings", methods=["POST"])
def add_ratings(user_id):
    file = request.files["ratings_file"]
    ratings = []
    for line in file:
        rating_data = json.loads(line.decode("utf-8"))
        ratings.append(rating_data)
    recommendation_engine.add_ratings(user_id, ratings)
    return "Le modèle de prédiction a été recalculé"

@main.route("/<int:user_id>/ratings/<int:movie_id>", methods=["GET"])
def movie_ratings(user_id, movie_id):
    rating = recommendation_engine.predict_rating(user_id, movie_id)
    if rating == -1:
        return "Aucune prédiction disponible pour ce film et cet utilisateur"
    else:
        return str(rating)

@main.route("/<int:user_id>/recommendations", methods=["GET"])
def get_recommendations(user_id):
    nb_movies = request.args.get("nb_movies", default=10, type=int)
    recommendations = recommendation_engine.recommend_for_user(user_id, nb_movies)
    return jsonify(recommendations)

@main.route("/ratings/<int:user_id>", methods=["GET"])
def get_ratings_for_user(user_id):
    ratings = recommendation_engine.get_ratings_for_user(user_id)
    return jsonify(ratings)

def create_app(spark_context, "Spark-movie-recommendation-main/app/ml-latest/movies.csv", "Spark-movie-recommendation-main/app/ml-latest/ratings.csv"):
    findspark.init()
    app = Flask(__name__)
    app.register_blueprint(main)

    recommendation_engine = RecommendationEngine(spark_context, "Spark-movie-recommendation-main/app/ml-latest/movies.csv", "Spark-movie-recommendation-main/app/ml-latest/ratings.csv")

    return app