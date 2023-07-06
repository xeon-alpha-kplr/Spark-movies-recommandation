import time
import sys
import cherrypy
import os
from cheroot.wsgi import Server as WSGIServer
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from app import create_app

# Importez les autres bibliothèques nécessaires

conf = SparkConf().setAppName("movie_recommendation-server")

sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])

"Spark-movie-recommendation-main/app/ml-latest/ratings.csv" = sys.argv[1] if len(sys.argv) > 1 else ""
"Spark-movie-recommendation-main/app/ml-latest/ratings.csv" = sys.argv[2] if len(sys.argv) > 2 else ""

app = create_app(sc, "Spark-movie-recommendation-main/app/ml-latest/ratings.csv", "Spark-movie-recommendation-main/app/ml-latest/ratings.csv")

cherrypy.tree.graft(app.wsgi_app, '/')
cherrypy.config.update({
    'server.socket_host': '0.0.0.0',
    'server.socket_port': 5000,
    'engine.autoreload.on': False
})

cherrypy.engine.start()
