from pyspark.ml.feature import CountVectorizer, Tokenizer, StringIndexer, VectorAssembler, Tokenizer, OneHotEncoder, Word2Vec, HashingTF, IndexToString
from pyspark.ml.linalg import SparseVector, Vectors
import numpy as np
import sparknlp
from pyspark.ml import Pipeline, PipelineModel
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, StopWordsCleaner, WordEmbeddingsModel, SentenceEmbeddings, BertEmbeddings, Word2VecModel
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT, DenseVector
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f


def preprocess_data(profiles):
    profiles = profiles.withColumn("about", f.when(f.col("about").isNull(), "").otherwise(f.col("about")))
    profiles = profiles.withColumn("position", f.col("position").cast("string"))
    profiles = profiles.withColumn("position", f.when(f.col("position").isNull(), "").otherwise(f.col("position")))
    profiles = profiles.withColumn("about_position", f.concat_ws(" ", f.col("about"), f.col("position")))
    profiles = profiles.withColumn("about_position", f.when(f.col("about_position") == " ", "No Info").otherwise(f.col("about_position")))\
        .select('about', 'position', 'education', 'experience', 'languages', 'followers', 'recommendations_count', 'about_position')

    # Preprocess 'about' and 'position' using Spark NLP
    document_assembler = DocumentAssembler() \
        .setInputCol("about_position") \
        .setOutputCol("ap_document")

    tokenizer = Tokenizer() \
        .setInputCols(["ap_document"]) \
        .setOutputCol("ap_token")

    stopwords_cleaner = StopWordsCleaner() \
        .setInputCols(["ap_token"]) \
        .setOutputCol("ap_clean_tokens")

    embeddings = BertEmbeddings.pretrained("small_bert_L2_128") \
        .setInputCols(["ap_document", "ap_clean_tokens"]) \
        .setOutputCol("ap_embeddings_bert")

    sentence_embeddings = SentenceEmbeddings() \
        .setInputCols(["ap_document", "ap_embeddings_bert"]) \
        .setOutputCol("about_position_embeddings")

    nlp_pipeline_about = Pipeline(stages=[document_assembler, tokenizer, stopwords_cleaner, embeddings, sentence_embeddings])

    # Apply NLP Pipeline
    nlp_model_about = nlp_pipeline_about.fit(profiles)
    processed_data1 = nlp_model_about.transform(profiles)

    processed_data = processed_data1 \
    .withColumn("num_education", f.when(f.size(f.col('education')).isNull(), 0).otherwise(f.size(f.col('education')))) \
    .withColumn("num_experience", f.when(f.size(f.col('experience')).isNull(), 0).otherwise(f.size(f.col('experience')))) \
    .withColumn("num_languages", f.when(f.size(f.col('languages')).isNull(), 0).otherwise(f.size(f.col('languages')))) \
    .withColumn("total_followers", f.when(f.col("followers").isNull(), 0).otherwise(f.col("followers"))) \
    .withColumn("num_recommendations", f.when(f.col("recommendations_count").isNull(), 0).otherwise(f.col("recommendations_count")))

    to_dense_udf = udf(lambda x: to_dense_vector(x), VectorUDT())

    
    processed_data = processed_data.withColumn(
        "about_position_embeddings_dense", 
        to_dense_udf(f.expr("about_position_embeddings.embeddings[0]"))
    )

    assembler = VectorAssembler(inputCols=[
    "about_position_embeddings_dense", "num_education", "num_experience", "num_languages",
    "total_followers", "num_recommendations",
    ], outputCol="features")

    final_data = assembler.transform(processed_data)

    return final_data

def to_dense_vector(embeddings_array):
    return Vectors.dense(embeddings_array)

