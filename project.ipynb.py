# Databricks notebook source
# MAGIC %md
# MAGIC # Final Project - Data Collection Lab (0940290)
# MAGIC ### Lihi Kaspi (214676140), Harel Oved (326042389) & Lior Zaphir (326482213)

# COMMAND ----------

from pyspark.sql.types import *
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import pandas as pd
from pyspark.ml.feature import CountVectorizer, Tokenizer, StringIndexer, VectorAssembler, Tokenizer, OneHotEncoder, Word2Vec, HashingTF, IndexToString
from pyspark.ml.linalg import SparseVector, Vectors
import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.window import Window
from datetime import datetime
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Relevant Data and Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ## i'm moving all the code cells that create a parquet file to different notebooks so we don't have to skip cells when running this notebook

# COMMAND ----------

# original datasets
companies = spark.read.parquet('/dbfs/linkedin_train_data')
profiles = spark.read.parquet('/dbfs/linkedin_people_train_data')

# COMMAND ----------

# new df of profiled with their "good profile" score -- code can be found in "Profile Score Calculation"
profiles_with_scores = spark.read.parquet("profiles_with_scores.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Pre-Processing

# COMMAND ----------

# MAGIC %md
# MAGIC # when you're done move the imports to the top !!!!!!!!!!!! OK!!!!!!!

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, Tokenizer, StringIndexer, VectorAssembler, Tokenizer, OneHotEncoder, Word2Vec, HashingTF, IndexToString
from pyspark.ml.linalg import SparseVector, Vectors
import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, StopWordsCleaner, WordEmbeddingsModel, SentenceEmbeddings
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT, DenseVector
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f

# COMMAND ----------

train_df, test_df = profiles_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

from sparknlp.pretrained import ResourceDownloader
print(ResourceDownloader.showPublicModels("word_embedding"))

# COMMAND ----------

import sparknlp
print(sparknlp.version())

# COMMAND ----------

# 1. Preprocess `about` using Spark NLP
document_assembler = DocumentAssembler() \
    .setInputCol("about") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

stopwords_cleaner = StopWordsCleaner() \
    .setInputCols(["token"]) \
    .setOutputCol("clean_tokens")

embeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols(["document", "clean_tokens"]) \
    .setOutputCol("embeddings")

sentence_embeddings = SentenceEmbeddings() \
    .setInputCols(["document", "embeddings"]) \
    .setOutputCol("about_embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stopwords_cleaner, embeddings, sentence_embeddings])

# Apply NLP Pipeline
nlp_model = nlp_pipeline.fit(train_df)
processed_data = nlp_model.transform(train_df)

# COMMAND ----------

# 2. Numerical Features
processed_data = processed_data.withColumn("num_education", f.size(f.col("education"))) \
    .withColumn("num_experience", f.size(f.col("experience"))) \
    .withColumn("num_languages", f.size(f.col("languages"))) \
    .withColumn("total_followers", f.col("followers")) \
    .withColumn("recommendations", f.col("recommendations_count"))

# COMMAND ----------

# 3. Categorical Features - TODO USE THEM???
# Index and encode categorical columns
# indexer = StringIndexer(inputCol="country_code", outputCol="country_code_index")
# encoder = OneHotEncoder(inputCol="country_code_index", outputCol="country_code_vec")
# cat_pipeline = Pipeline(stages=[indexer, encoder])

# Fit and transform categorical pipeline
# cat_model = cat_pipeline.fit(processed_data)
# processed_data = cat_model.transform(processed_data)

# COMMAND ----------

# 4. Assemble features
# assembler = VectorAssembler(inputCols=[
#     "about_embeddings", "num_education", "num_experience", "num_languages",
#     "total_followers", "recommendations", "country_code_vec"
# ], outputCol="features")

# COMMAND ----------

# from pyspark.sql.functions import udf
# from pyspark.ml.linalg import VectorUDT, DenseVector
# from pyspark.ml.feature import VectorAssembler
# import pyspark.sql.functions as f

# # UDF to extract embeddings and convert to DenseVector
# def extract_embeddings(embeddings):
#     return DenseVector(embeddings[0].embeddings) if embeddings else DenseVector([])

# extract_embeddings_udf = udf(extract_embeddings, VectorUDT())

# # Apply the UDF to extract embeddings
# processed_data = processed_data.withColumn("about_embeddings_vector", extract_embeddings_udf(f.col("about_embeddings")))

# # Assemble features
# assembler = VectorAssembler(
#     inputCols=[
#         "about_embeddings_vector", "num_education", "num_experience", "num_languages",
#         "total_followers", "recommendations"
#     ],
#     outputCol="features"
# )

# final_data = assembler.transform(processed_data)

# # Select relevant columns
# final_data = final_data.select("features", "filled_precent")

# # Save processed data
# # final_data.write.parquet("processed_profile_data.parquet")
# display(final_data.limit(100))

# COMMAND ----------

display(processed_data.limit(100))

# COMMAND ----------

def to_dense_vector(embeddings_array):
    return Vectors.dense(embeddings_array)

# Register a UDF to convert arrays to dense vectors
to_dense_udf = udf(lambda x: to_dense_vector(x), VectorUDT())

# Apply the UDF to the embeddings column (adjust column name as needed)
processed_data = processed_data.withColumn(
    "about_embeddings_dense", 
    to_dense_udf(f.expr("about_embeddings.embeddings[0]"))
)


# Assemble features
assembler = VectorAssembler(inputCols=[
    "about_embeddings_dense", "num_education", "num_experience", "num_languages",
    "total_followers", "recommendations",
], outputCol="features", handleInvalid="skip")

final_data = assembler.transform(processed_data)

# Select relevant columns
final_data = final_data.select("features", "filled_precent")

# Save processed data
# final_data.write.parquet("processed_profile_data.parquet")
display(final_data.limit(100))

# COMMAND ----------

# final_data.write.parquet("processed_profile_data.parquet")
final_data.write.parquet("/Workspace/Users/harel.oved@campus.technion.ac.il/processed_data.parquet")

# COMMAND ----------

final_data = spark.read.parquet('/Workspace/Users/harel.oved@campus.technion.ac.il/processed_data.parquet')

# COMMAND ----------

import matplotlib.pyplot as plt

sample = profiles_df.select('profile_score').sample(False, 0.1).toPandas()

plt.figure(figsize=(10, 6))
plt.hist(sample['profile_score'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of Profile Scores')
plt.xlabel('Profile Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scraped Data Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC #### Job Titles and Locations

# COMMAND ----------

jobs = profiles.select('name', 'id', 'city', 'country_code', f.col('current_company').getField('name').alias('company_name'), f.col('experience')[0].getField('title').alias('job_title'), 'position')

# COMMAND ----------

jobs.display(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Clustering job titles into meta job titles

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, split
from pyspark.sql.types import StringType
from pyspark.ml.feature import Word2Vec, Tokenizer

# Create a DataFrame with the specified centroids
centroids_data = [
    ('Leadership',), ('Product',), ('Engineering',), ('DataScience',), ('Operations',),
    ('Marketing',), ('Sales',), ('Design',), ('Support',), ('Finance',),
    ('Resources',), ('Research',), ('Healthcare',), ('Education',), ('Security',),
    ('Logistics',), ('Legal',), ('Quality',), ('Management',), ('Content',)
]

centroids_df = spark.createDataFrame(centroids_data, ['processed_title'])

# Preprocess job titles
job_titles_df = jobs.select(
    when(col('job_title').isNotNull(), lower(col('job_title')))
    .otherwise(lower(col('position')))
    .alias('processed_title')
)
job_titles_df = job_titles_df.dropna()
tokenizer = Tokenizer(inputCol="processed_title", outputCol="tokened_title")
w2v = Word2Vec(inputCol="tokened_title", outputCol="vector", vectorSize=200, minCount=1)




# Build the pipeline
pipeline = Pipeline(stages=[tokenizer, w2v])

# Train the pipeline model
model_vectorize = pipeline.fit(job_titles_df)



# Create embeddings for job titles and centroids
jobs_with_vectors = model_vectorize.transform(job_titles_df)
centroids_with_vectors = model_vectorize.transform(centroids_df)


jobs_temp = jobs_with_vectors.withColumnRenamed('vector', 'job_vector')
jobs_temp = jobs_temp.withColumnRenamed('processed_title', 'job_title')

centroids_temp = centroids_with_vectors.withColumnRenamed('processed_title', 'meta_job')
centroids_temp = centroids_temp.withColumnRenamed('vector', 'centroid_vector')

joined = jobs_temp.join(centroids_temp)
display(joined.limit(10))

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import DenseVector
import math

# Define a function to calculate cosine similarity
def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return None
    dot_product = float(v1.dot(v2))  # Dot product of the two vectors
    norm_v1 = math.sqrt(v1.dot(v1))  # Magnitude (norm) of v1
    norm_v2 = math.sqrt(v2.dot(v2))  # Magnitude (norm) of v2
    if norm_v1 == 0 or norm_v2 == 0:
        return None  # Avoid division by zero
    return dot_product / (norm_v1 * norm_v2)

# Register the function as a UDF
cosine_similarity_udf = udf(cosine_similarity, StringType())

# Add a new column to compute cosine similarity
joined = joined.withColumn(
    "cosine_similarity",
    cosine_similarity_udf(col("job_vector"), col("centroid_vector"))
)

# Show the result
joined.display()


# COMMAND ----------

window_spec = Window.partitionBy("job_title").orderBy(col("cosine_similarity").desc())

# Rank centroids for each job and select the closest one
ranked_df = joined.withColumn("rank", f.row_number().over(window_spec))

# Filter for the closest centroid
closest_centroids = ranked_df.filter(col("rank") == 1)

# Select relevant columns
result_df = closest_centroids.select(
    col("job_title"),
    col("meta_job").alias("closest_centroid"),
    col("cosine_similarity")
)

# Display the result
result_df.display()

# COMMAND ----------

profiles_with_state = profiles.withColumn(
    "state",
    split(col("city"), ", ")[1]  # The second element is the state
)

# Show the results
states_df = profiles_with_state.select("state").dropDuplicates().dropna()
states_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Scraping 

# COMMAND ----------

# MAGIC %pip install selenium
# MAGIC %pip install beautifulsoup4

# COMMAND ----------

# MAGIC %md
# MAGIC ## this notebook is way to big so maybe move the scraping process to a new notebook and save the data in a parquet file to read from this notebook?
# MAGIC Yes we should divide the parts to differnet files it will be good to display them separately in the git
# MAGIC
# MAGIC exactly

# COMMAND ----------

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
from bs4 import BeautifulSoup

def get_html(job, state):
    print(state, job)
    # Define the job title and location
    paginaton_url = 'https://www.indeed.com/jobs?q={}&l={}&'
    driver.get(paginaton_url.format(job, state))
    time.sleep(random.randint(2, 6))
    return driver.page_source

def get_count(html):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Find the job results container
    job_count_div = soup.find("div", class_="jobsearch-JobCountAndSortPane-jobCount")

    count = job_count_div.find("span").text.split(' ')[0][:-1]
    return count

centroids_data = [
    ('Leadership',), ('Product',), ('Engineering',), ('DataScience',), ('Operations',),
    ('Marketing',), ('Sales',), ('Design',), ('Support',), ('Finance',),
    ('Resources',), ('Research',), ('Healthcare',), ('Education',), ('Security',),
    ('Logistics',), ('Legal',), ('Quality',), ('Management',), ('Content',)
]

centroids_df = spark.createDataFrame(centroids_data, ['processed_title'])
job_state_df = states_df.join(centroids_df)

df = job_state_df.toPandas()

options = webdriver.ChromeOptions()
options.add_argument("--user-data-dir=/tmp/chrome_user_data")

driver = webdriver.Chrome(options=options)

for index, row in df.iterrows():
    state = row['state']
    job = row['processed_title']
    html = get_html(job, state)
    count = get_count(html)
    print(count)

driver.quit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Good Profiles Model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### i want to predit a numeric score and not binary label -- will be better for the final stage of suggesting improvemnts

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training the Model

# COMMAND ----------

# MAGIC %md
# MAGIC possible models:
# MAGIC - Decision Tree Regressor
# MAGIC - Random Forest Regressor
# MAGIC - Gradient-Boosted Trees Regressor
# MAGIC
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluating the model

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC when checking accuracy - accepted score should be between (real_score-5, real_score+5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profile Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC ### 'about' Section Optimization

# COMMAND ----------

# take: about (if not null), position, job title, reccomendations 
# --> return: a sentence or two describing the person and job (in a new column called 'new_about')
# if all null: return message 'could not generate a short bio -- add more information to your profile' (put null in 'new_about' and add message in a new column called 'about_message')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improvements and Suggetions

# COMMAND ----------

# MAGIC %md
# MAGIC score ranking:
# MAGIC - excellent score - 90+ and no suggestions
# MAGIC - high score - 90+ and atleast one suggestion
# MAGIC - medium high score - 60-90
# MAGIC - medium score - 40-60
# MAGIC - medium low score - 20-40
# MAGIC - low score - 20>

# COMMAND ----------

score_messages = {
    'excellent_score': 'Your profile is excellent, keep it up!',
    'high_score': 'Your profile is very strong, Check the suggestions to make it excellent',
    'medium_high_score': 'Your profile is good, Try to follow the suggestions to make it even better',
    'medium_score': 'Your profile could use a few improvements, Try to follow the suggestions to make it even better',
    'medium_low_score': 'Your profile needs to improve, Try to follow the suggestion to make it better',
    'low_score': 'Your profile is weak, Try to follow the suggestion to make it better',
}

# COMMAND ----------

missing_field_messages = {
    'no_experience': 'Add previous/current comapnies you worked in', 
    'no_education': 'List your degrees and schools you graduated from',
    'no_about': 'Add a short bio about yourself, here is a suggestion: ',
    'suggested_about': 'Try out this about section: ',
    'no_company': 'Add the compant you currently work in',
    'no_languages': 'List all the languages you know and the level of knowledge',
    'no_position': 'Add the position you are currently in',
    'no_posts': 'Try to be more active with you account',
    'no_recommendations': 'Ask a colleague to write a few words about you',
    'missing_experience': 'There is a gap in your resume, Don\'t forget to add all of the previous comapnies you worked in',
    'low_followers': 'Ask your colleagues and friends to follow you on LinkedIn!'
    }

# COMMAND ----------

# placeholder name for the predictions: predicted_df (has all the previous columns + score predictions)

predicted_df = predicted_df.withColumn(
  'score_rank', 
  f.when(f.col('score') < 20, 'low_score'
  ).when(f.col('score') < 40, 'medium_low_score'
  ).when(f.col('score') < 60, 'medium_score'
  ).when(f.col('score') < 90, 'medium_high_score'
  ).when(f.col('filled_percent') < 100, 'high_score'
  ).otherwise('excellent_score')
)

predicted_df = predicted_df.withColumn(
  'score_message',
  score_messages.get(f.col('score_rank'))
)

# COMMAND ----------

# find if there are gaps in the experience array (name new column: 'gap_in_experience')
# TODO: Binary or explicit time period? 

# COMMAND ----------

predicted_df = predicted_df.withColumn('suggestions', f.array())

predicted_df = predicted_df.withColumn(
  'suggestions',
  f.when(
    f.size(f.col('education')) == 0, 
    f.concat('suggesstions', f.array(missing_field_messages.get('no_education')))
  ).when(
    f.size(f.col('current_company')) == 0, 
    f.concat('suggesstions', f.array(missing_field_messages.get('no_company')))
  ).when(
    f.size(f.col('languages')) == 0, 
    f.concat('suggesstions', f.array(missing_field_messages.get('no_languages')))
  ).when(
    f.size(f.col('posts')) == 0, 
    f.concat('suggesstions', f.array(missing_field_messages.get('no_posts')))
  ).when(
    f.col('recommendations_count') == 0, 
    f.concat('suggesstions', f.array(missing_field_messages.get('no_recommendations')))
  ).when(
    f.col('about').isNull() & f.col('new_about').isNotNull(), 
    f.concat('suggesstions', f.array(missing_field_messages.get('no_about') + f.col('new_about')))
  ).when(
    f.col('about').isNotNull() & f.col('new_about').isNotNull(), 
    f.concat('suggesstions', f.array(missing_field_messages.get('suggested_about') + f.col('new_about')))
  ).when(
    f.col('about_message').isNotNull(), 
    f.concat('suggesstions', f.array(f.col('about_message')))
  ).when(
    f.col('position').isNull(),
    f.concat('suggesstions', f.array(missing_field_messages.get('no_position')))
  ).when(
    f.col('followers') < 20,
    f.concat('suggesstions', f.array(missing_field_messages.get('low_followers')))
  ).when(
    f.size(f.col('experience')) == 0, 
    f.concat('suggesstions', f.array(missing_field_messages.get('no_experience')))
  ).when(
    f.col('gap_in_experience').isNotNull(), # TODO: adapt to binary or time period
    f.concat('suggesstions', f.array(missing_field_messages.get('missing_experience'))
  ).otherwise(f.col('suggestions'))
)

# COMMAND ----------

# df_with_array_matches = df.withColumn(
#     "all_matches",
#     array(
#         when(df["name"] == "Alice", lit("Match 1")),
#         when(df["name"].startswith("A"), lit("Match 2"))
#     )
# )

# COMMAND ----------

optemized_df = predicted_df.select('name', 'id', 'url', 'score_rank', 'score_message', 'suggestions')
display(optemized_df)
