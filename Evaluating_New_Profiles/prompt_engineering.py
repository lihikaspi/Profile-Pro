# Databricks notebook source
from pyspark.sql.types import *
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import pandas as pd
from pyspark.ml.feature import CountVectorizer, Tokenizer as Tokenizer_feature, StringIndexer, VectorAssembler, OneHotEncoder, Word2Vec, HashingTF, IndexToString
from pyspark.ml.linalg import SparseVector, Vectors
import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.window import Window
from datetime import datetime
import re
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, concat_ws, udf, lit
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.ml.functions import vector_to_array
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from huggingface_hub import InferenceClient
from huggingface_hub import login
import time
from pyspark.ml.linalg import VectorUDT, DenseVector
import numpy as np
import sparknlp
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, StopWordsCleaner, WordEmbeddingsModel, SentenceEmbeddings, BertEmbeddings, Word2VecModel
from pyspark.ml.classification import MultilayerPerceptronClassificationModel

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre process good profile data

# COMMAND ----------

def strip_and_choose_first(str_lst):
    return str_lst.strip("[]").split(", ")[0]


def process_education(degree, field, title):
    # Extract degree, field, and school title from each education entry
    degree = strip_and_choose_first(degree)
    field = strip_and_choose_first(field)
    title = strip_and_choose_first(title)
    edu_details = f"{degree} in {field} from {title}"
    return edu_details


def preprocess_profiles(df):
    """
    df: profiles dataframe, such that name, id, city,  experience, position are in the correct format
    returns pre processed dataframe.
    """
    jobs = df.select('name', 'id', 'city', f.col('experience')[0].getField('title').alias('job_title'), 'position')
    process_education_udf = udf(process_education, StringType())
    job_titles_df = jobs.select(
        f.when(f.col('job_title').isNotNull(), f.lower(f.col('job_title')))
        .otherwise(f.when(f.col('position').isNotNull(), f.lower(f.col('position'))).otherwise(f.lit('')))
        .alias('processed_title'), 'id'
    )

    df = df.join(job_titles_df, on='id')
    edu_filtered_df = df.filter((col("education").isNotNull()) & (col("education") != f.lit([])))
    no_edu_df = df.filter((col("education").isNull()) | (col("education") == f.lit([])))

    filtered_df = edu_filtered_df.withColumn('degree', col('education').getField('degree').cast('string'))
    filtered_df = filtered_df.withColumn('field', col('education').getField('field').cast('string'))
    filtered_df = filtered_df.withColumn('school', col('education').getField('title').cast('string'))

    # Process the DataFrame
    edu_filtered_df = filtered_df.withColumn("processed_education", 
                                            process_education_udf(col('degree'), col('field'), col('school')))
    no_edu_df = no_edu_df.withColumn("processed_education", lit(''))
    edu_filtered_df = edu_filtered_df.select(['id', 'processed_education', 'processed_title', 'name','city'])
    no_edu_df = no_edu_df.select(['id', 'processed_education', 'processed_title', 'name','city'])
    df = edu_filtered_df.union(no_edu_df)
    return df 

def generate_small_good_sample(spark):
    profiles_with_scores = spark.read.parquet("/Workspace/Users/lihi.kaspi@campus.technion.ac.il/user_profiles_with_scores.parquet")
    profiles = spark.read.parquet('/dbfs/linkedin_people_train_data')
    profiles_with_scores = profiles_with_scores.withColumn(
        'label', 
        f.when(f.col('profile_score') < 5, 0
        ).when(f.col('profile_score') < 10, 1
        ).when(f.col('profile_score') < 15, 2
        ).when(f.col('profile_score') < 20, 3
        ).otherwise(4)
    )
    df = preprocess_profiles(profiles_with_scores)
    df = df.join(profiles_with_scores, on='id')
    good_profiles_df = df.filter(col('label').isin([3,4])).select(['id','processed_education','processed_title', 'about'])
    good_profiles_df = good_profiles_df.limit(10000)
    good_profiles_df.write.mode("overwrite").parquet("/Workspace/Users/lihi.kaspi@campus.technion.ac.il/sample_good_profile_data.parquet")

# COMMAND ----------

def df_to_vector(df, good_profiles_df):
    tokenizer_title = Tokenizer_feature(inputCol="processed_title", outputCol="tokened_title")
    w2v_title = Word2Vec(inputCol="tokened_title", outputCol="vector_title", vectorSize=200, minCount=1)

    tokenizer_edu = Tokenizer_feature(inputCol="processed_education", outputCol="tokened_edu")
    w2v_edu = Word2Vec(inputCol="tokened_edu", outputCol="vector_edu", vectorSize=200, minCount=1)

    pipeline = Pipeline(stages=[tokenizer_title, w2v_title, tokenizer_edu, w2v_edu])

    model_vectorize = pipeline.fit(df)

    # Create embeddings for job titles and centroids
    df_with_vectors = model_vectorize.transform(df)
    good_with_vectors = model_vectorize.transform(good_profiles_df)
    return df_with_vectors, good_with_vectors

# COMMAND ----------

def cross_dfs(df_with_vectors, good_with_vectors):
    profiles = df_with_vectors.withColumnRenamed("vector_title", "pos_embed") \
                            .withColumnRenamed("vector_edu", "edu_embed") \
                            .withColumnRenamed("id", "profiles_id")

    good_profiles = good_with_vectors.withColumnRenamed("vector_title", "pos_embed_good") \
                                    .withColumnRenamed("vector_edu", "edu_embed_good") \
                                    .withColumnRenamed("id", "good_profile_id")

    good_profiles = good_profiles.select(["good_profile_id", "pos_embed_good", "edu_embed_good"])
    profiles = profiles.select(["profiles_id", "pos_embed", "edu_embed"])

    profiles = profiles.withColumn("edu_embed", vector_to_array(col("edu_embed")))
    profiles = profiles.withColumn("pos_embed", vector_to_array(col("pos_embed")))

    good_profiles = good_profiles.withColumn("edu_embed_good", vector_to_array(col("edu_embed_good")))
    good_profiles = good_profiles.withColumn("pos_embed_good", vector_to_array(col("pos_embed_good")))

    good_profiles_broadcast = broadcast(good_profiles)

    profiles_cross = profiles.join(good_profiles_broadcast, how="inner")
    return profiles_cross

# COMMAND ----------


def dot_product(vec1, vec2):
    return F.expr(f"""
        aggregate(transform({vec1}, (x, i) -> x * {vec2}[i]), 0D, (acc, x) -> acc + x)
    """)

def vector_norm(vec):
    return F.sqrt(F.expr(f"aggregate(transform({vec}, x -> x * x), 0D, (acc, x) -> acc + x)"))
def compute_sim(profiles_cross):
    edu_dot_product = dot_product("edu_embed", "edu_embed_good")
    pos_dot_product = dot_product("pos_embed", "pos_embed_good")


    edu_norm_profile = vector_norm("edu_embed")
    edu_norm_good = vector_norm("edu_embed_good")

    pos_norm_profile = vector_norm("pos_embed")
    pos_norm_good = vector_norm("pos_embed_good")

    profiles_cross = profiles_cross.withColumn(
        "edu_sim", edu_dot_product / (edu_norm_profile * edu_norm_good)
    ).withColumn(
        "pos_sim", pos_dot_product / (pos_norm_profile * pos_norm_good)
    ).withColumn(
        "total_sim", F.col("edu_sim") + F.col("pos_sim")
    )
    return profiles_cross

# COMMAND ----------


def get_best_matches(profiles_cross):
    # order by highest similarity
    window_spec = Window.partitionBy("profiles_id").orderBy(col("total_sim").desc())
    # Rank the matches and filter to keep only the best match per profile
    best_matches = profiles_cross.withColumn("rank", row_number().over(window_spec)).filter(col("rank") == 1)

    best_matches = best_matches.select(
        col("profiles_id"),
        col("good_profile_id").alias("matched_good_profile_id"),
        col("total_sim")
    )
    return best_matches

# COMMAND ----------

def get_match_df(best_matches, spark):
    good_profiles = spark.read.parquet("/Workspace/Users/lihi.kaspi@campus.technion.ac.il/sample_good_profile_data.parquet")
    
    good_profiles = good_profiles.select(['id','about']).dropna().withColumnRenamed('id', "matched_good_profile_id")

    match_df = best_matches.join(good_profiles, on="matched_good_profile_id")
    return match_df

# COMMAND ----------

def generate_sections(bad_profile_df, match_df, spark):
    access_token = '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'
    login(access_token)


    df = match_df.withColumnRenamed('profiles_id', 'id').join(bad_profile_df.withColumnRenamed('some_column_name', 'id'), on="id")
    pd_df = df.toPandas()
    
    def create_section(user_data, procesed_edu, city, name, proccesed_title):
        client = InferenceClient(token=access_token)
        input_prompt = f"This is an about section of a user similar to me:{user_data}. build an about section for me. my name is {name}, I live in {city}. my education details are {procesed_edu} and my job title is {proccesed_title}.  Do not use things like [Assuming a similar role as Fleet Account Manager based on Josh's profession] Business Development Specialist at [Assuming a company similar to Knapheide Manufacturing], it should look like a real about section"
        completion = client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.3", 
            prompt=input_prompt, 
            max_new_tokens=500
        )
        return completion
    i = 0
    abouts = []
    for _, row in pd_df.iterrows():
        print(i)
        i+=1
        user_data = row['about']
        name  = row['name']
        city = row['city']
        proccesed_edu = row['processed_education']
        proccesed_title = row['processed_title']
        completion = create_section(user_data, proccesed_edu, city, name, proccesed_title)
        time.sleep(2)
        abouts.append((row["id"],completion))
        if i == 400:
            break
    generated_abouts_df = spark.createDataFrame(abouts, ["id", "about"])

    return generated_abouts_df

# COMMAND ----------

def optimize_profiles(df,spark, good_profiles=None):
    """
    Optimize the profiles based on the good profiles provided.
    If no good_profiles are provided, default to using the sample good profiles we defined.
    """
    if good_profiles is None:
        good_profiles = spark.read.parquet("/Workspace/Users/lihi.kaspi@campus.technion.ac.il/sample_good_profile_data.parquet")
    df = preprocess_profiles(df)
    df_vector, good_profiles_vector = df_to_vector(df, good_profiles)
    profiles_cross = cross_dfs(df_vector, good_profiles_vector)
    sim_cross = compute_sim(profiles_cross)
    best_matches = get_best_matches(sim_cross)
    match_df = get_match_df(best_matches, spark)
    gen_df = generate_sections(df, match_df, spark)
    return gen_df
