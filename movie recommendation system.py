# Databricks notebook source
# MAGIC %md 
# MAGIC ### Spark HW3 Moive Recommendation
# MAGIC In this notebook, we will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/)

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# COMMAND ----------

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part1: Data ETL and Data Exploration

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

movies = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)

# COMMAND ----------

movies.show(5)

# COMMAND ----------

ratings.show(5)

# COMMAND ----------

tmp1 = ratings.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

# COMMAND ----------

tmp1 = sum(ratings.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Spark SQL and OLAP 

# COMMAND ----------

# MAGIC %md ### The number of Users

# COMMAND ----------

tmp_q1 = ratings.select('userid').distinct().count()
print ('There totally have {} users'.format(tmp_q1))

# COMMAND ----------

# MAGIC %md ### The number of Movies

# COMMAND ----------

tmp_q2 = movies.select('movieid').distinct().count()
print ('There totally have {} movies'.format(tmp_q2))

# COMMAND ----------

# MAGIC %md ### How many movies are rated by users? List movies not rated before

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.functions import isnan
tmp_q3 = ratings.select('movieid').distinct().count()
print ('{} movies have not been rated'.format(tmp_q2-tmp_q3))
tmp_q3 = movies.join(ratings, 'movieid', "leftouter") 
tmp_q3.select('movieid', 'title', 'rating').where(col('rating').isNull()).show()

# COMMAND ----------

# MAGIC %md ### List Movie Genres

# COMMAND ----------

import pyspark.sql.functions as f
genres = movies.select(f.split(movies['genres'], '\|')).collect()

hashset = set()
for row in genres:
  for ele in row[0]:
    hashset.add(ele)
print (hashset), len(hashset)

# COMMAND ----------

# MAGIC %md ### Movie for Each Category

# COMMAND ----------

import pyspark.sql.functions as f
genres = movies.select(f.split(movies['genres'], '\|')).collect()
print len(genres)
df, row_i, col_i = [[0 for i in xrange(len(hashset))] for j in xrange(len(genres))], 0, 0
print len(df), len(df[0])
for genre in hashset:
  row_i = 0
  for row in genres:
      if genre in row[0]:
        df[row_i][col_i] = True
      else:
        df[row_i][col_i] = None     
      row_i += 1 
  col_i += 1
q5 = movies.select("movieid","title")
#q5.show()

# COMMAND ----------

from pandas import DataFrame
from pyspark.sql.functions import monotonically_increasing_id

tmp = DataFrame.from_records(df)
tmp = spark.createDataFrame(tmp,schema=list(hashset))

q5 = q5.withColumn("id", monotonically_increasing_id())
tmp = tmp.withColumn("id", monotonically_increasing_id())

tmp_q5 = q5.join(tmp, 'id').drop('id')
#here list all categorys
tmp_q5.show()

# COMMAND ----------

#here list "Drama" movies as an example
tmp_drama = tmp_q5.select("movieid", "title").where(col("Drama").isNotNull())
print ("{} movies are drama, there are:".format(tmp_drama.count()))
tmp_drama.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part2: Spark ALS based approach for training model
# MAGIC We will use an RDD-based API from [pyspark.mllib](https://spark.apache.org/docs/2.1.1/mllib-collaborative-filtering.html) to predict the ratings, so let's reload "ratings.csv" using ``sc.textFile`` and then convert it to the form of (user, item, rating) tuples.

# COMMAND ----------

movie_rating = sc.textFile("/FileStore/tables/ratings.csv")

# COMMAND ----------

header = movie_rating.take(1)[0]
rating_data = movie_rating.filter(lambda line: line!=header).map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

# COMMAND ----------

# check three rows
rating_data.take(3)

# COMMAND ----------

# MAGIC %md Now we split the data into training/validation/testing sets using a 6/2/2 ratio.

# COMMAND ----------

train, validation, test = rating_data.randomSplit([6,2,2],seed = 7856)

# COMMAND ----------

train.cache()

# COMMAND ----------

validation.cache()

# COMMAND ----------

test.cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ALS Model Selection and Evaluation
# MAGIC 
# MAGIC With the ALS model, we can use a grid search to find the optimal hyperparameters.

# COMMAND ----------

# DBTITLE 1,Transfer data from RDD to DF
from pyspark.sql.functions import col
train_DF = train.toDF(['userId','movieId','rating'])
train = train_DF.select(col('userId').cast('integer'), col('movieId').cast('integer'), col('rating').cast('float'))
print (type(train))
#train.show()

validation_DF = validation.toDF(['userId','movieId','rating'])
validation = validation_DF.select(col('userId').cast('integer'), col('movieId').cast('integer'), col('rating').cast('float'))

test_DF = test.toDF(['userId','movieId','rating'])
test = test_DF.select(col('userId').cast('integer'), col('movieId').cast('integer'), col('rating').cast('float'))

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
def train_ALS(train_data, validation_data, num_iters, reg_param, ranks):
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in reg_param:
            # 1) Build the recommendation model using ALS on the training data
            # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
            als = ALS(rank=rank, maxIter=num_iters, regParam=reg, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
            model = als.fit(train_data)
            
            # 2) Evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                            predictionCol="prediction")
            rmse_error = evaluator.evaluate(predictions)
            print("Root-mean-square error = " + str(rmse_error))            

            print ('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, rmse_error))
            if rmse_error < min_error:
                min_error = rmse_error
                best_rank = rank
                best_regularization = reg
                best_model = model
    print ('\nThe best model has {} latent factors and regularization = {}'.format(best_rank, best_regularization))
    return best_model

# COMMAND ----------

num_iterations = 10
ranks = [6, 8, 10, 12, 14]
reg_params = [0.05, 0.1, 0.2, 0.4, 0.8]
import time

start_time = time.time()
final_model = train_ALS(train, validation, num_iterations, reg_params, ranks)

print ('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

# COMMAND ----------

from sklearn.model_selection import learning_curve
#%matplotlib inline 
def plot_learning_curve(num_iters, train_data, validation_data, reg, rank):
    iter_num, rmse = [], []
    for iter_e in num_iters:
      # 1) Build the recommendation model using ALS on the training data
      # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
      als = ALS(rank=rank, maxIter=iter_e, regParam=reg, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
      model = als.fit(train_data)
            
      # 2) Evaluate the model by computing the RMSE on the validation data
      predictions = model.transform(validation_data)
      evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
      rmse_error = evaluator.evaluate(predictions)
      print("Root-mean-square error = " + str(rmse_error))            
      print ('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, rmse_error))
      iter_num.append(iter_e)
      rmse.append(rmse_error)
    # 3) plot curves    
    fig, ax = plt.subplots(figsize=(20,8))
    ax.plot(iter_num, rmse, 'k--')
    ax.plot(iter_num, rmse, 'ro')

    # set ticks and tick labels
    ax.set_xticks(iter_num)
    ax.set_xticklabels(iter_num)
    ax.set_yticks(rmse)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')   
    fig.suptitle("learning curves (ALS)")
    display(fig)
    
    return plt

# COMMAND ----------

iter_array = [1, 2, 5, 10]
plot_learning_curve(iter_array, train, validation, 0.2, 10)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model testing
# MAGIC And finally, make a prediction and check the testing error.

# COMMAND ----------

# Evaluate the model by computing the RMSE on the test data
predictions = final_model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = final_model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = final_model.recommendForAllItems(10)
#userRecs.show()
movieRecs.show()
