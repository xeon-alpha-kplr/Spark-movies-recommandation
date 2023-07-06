from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext

class RecommendationEngine:
    def __init__(self, sc, "/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/movies.csv", "Spark-movie-recommendation-main/app/ml-latest/ratings.csv"):
        self.spark = SparkSession(sc)
        self.spark.conf.set("spark.sql.shuffle.partitions", "4")
        
        self.load_movies("/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/movies.csv")
        self.load_ratings("Spark-movie-recommendation-main/app/ml-latest/ratings.csv")
        
        self.max_user_identifier = self.ratings_df.select(F.max("userId")).first()[0]
        if self.max_user_identifier is None:
            self.max_user_identifier = 0
        
        self.__train_model()
    
    def load_movies(self, "/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/movies.csv"):
        movies_schema = StructType([
            StructField("movieId", IntegerType(), True),
            StructField("title", StringType(), True),
            # Include other relevant fields
        ])
        
        self.movies_df = self.spark.read.csv("/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/movies.csv", header=True, schema=movies_schema)
    
    def load_ratings(self, "Spark-movie-recommendation-main/app/ml-latest/ratings.csv"):
        ratings_schema = StructType([
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", DoubleType(), True),
            # Include other relevant fields
        ])
        
        self.ratings_df = self.spark.read.csv("Spark-movie-recommendation-main/app/ml-latest/ratings.csv", header=True, schema=ratings_schema)
    
    def create_user(self, user_id=None):
        if user_id is None:
            user_id = self.max_user_identifier + 1
            self.max_user_identifier += 1
        
        return user_id
    
    def is_user_known(self, user_id):
        return user_id is not None and user_id <= self.max_user_identifier
    
    def get_movie(self, movie_id=None):
        if movie_id is None:
            sample_movie = self.movies_df.sample(0.1).limit(1)
            return sample_movie.select("movieId", "title")
        else:
            return self.movies_df.filter(F.col("movieId") == movie_id).select("movieId", "title")
    
    def get_ratings_for_user(self, user_id):
        return self.ratings_df.filter(F.col("userId") == user_id).select("movieId", "userId", "rating")
    
    def add_ratings(self, user_id, ratings):
        new_ratings_df = self.spark.createDataFrame(ratings, self.ratings_df.schema)
        self.ratings_df = self.ratings_df.union(new_ratings_df)
        
        train_df, test_df = self.ratings_df.randomSplit([0.8, 0.2])
        self.training = train_df.cache()
        self.test = test_df.cache()
        
        self.__train_model()
    
    def predict_rating(self, user_id, movie_id):
        rating_df = self.spark.createDataFrame([(user_id, movie_id)], ["userId", "movieId"])
        predictions = self.model.transform(rating_df).select("prediction").collect()
        
        if len(predictions) == 0:
            return -1
        else:
            return predictions[0][0]
    
    def recommend_for_user(self, user_id, nb_movies):
        user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
        recommendations = self.model.recommendForUserSubset(user_df, nb_movies).select("recommendations.movieId")
        movie_ids = [r[0] for r in recommendations.collect()]
        
        recommended_movies = self.movies_df.filter(F.col("movieId").isin(movie_ids))
        
        return recommended_movies.select("title", "genre") 
    
    def __train_model(self):
        als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
                  coldStartStrategy="drop")
        self.model = als.fit(self.training)
        self.__evaluate()

    def __evaluate(self):
        predictions = self.model.transform(self.test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        self.rmse = evaluator.evaluate(predictions)
        print(f"Root Mean Squared Error (RMSE): {self.rmse}")

    def __init__(self, sc, "/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/movies.csv", "Spark-movie-recommendation-main/app/ml-latest/ratings.csv"):
        self.spark = SparkSession(sc)
        self.spark.conf.set("spark.sql.shuffle.partitions", "4")

        self.load_movies("/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/movies.csv")
        self.load_ratings("Spark-movie-recommendation-main/app/ml-latest/ratings.csv")

        self.max_user_identifier = self.ratings_df.select(F.max("userId")).first()[0]
        if self.max_user_identifier is None:
            self.max_user_identifier = 0

        self.training, self.test = self.ratings_df.randomSplit([0.8, 0.2])
        self.__train_model()

# Création d'une instance de la classe RecommendationEngine
engine = RecommendationEngine(sc, "/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/movies.csv", "/workspaces/Spark-movies-recommandation/Spark-movie-recommendation-main/app/ml-latest/ratings.csv")

# Exemple d'utilisation des méthodes de la classe RecommendationEngine
user_id = engine.create_user(None)
if engine.is_user_known(user_id):
    movie = engine.get_movie(None)
    ratings = engine.get_ratings_for_user(user_id)
    engine.add_ratings(user_id, ratings)
    prediction = engine.predict_rating(user_id, movie.movieId)
    recommendations = engine.recommend_for_user(user_id, 10)
