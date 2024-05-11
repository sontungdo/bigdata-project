import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Build a SparkSession
spark = SparkSession.builder \
    .appName('Toxic Comment Classification') \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

# Read the input data from system arguments
input_dir = sys.argv[1]
train = spark.read.csv(input_dir + "/train.csv", header=True, inferSchema=True, sep=",", quote='"', escape='"', multiLine=True)
test = spark.read.csv(input_dir + "/test.csv", header=True, inferSchema=True, sep=",", quote='"', escape='"', multiLine=True)

# Local test
# input_dir = "../../data-comment"
# train = spark.read.csv(input_dir + "/train_small.csv", header=True, inferSchema=True, sep=",", quote='"', escape='"', multiLine=True)
# test = spark.read.csv(input_dir + "/test_small.csv", header=True, inferSchema=True, sep=",", quote='"', escape='"', multiLine=True)

out_cols = [col for col in train.columns if col not in ["id", "comment_text"]]

# Tokenize the text
tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
train_tokens = tokenizer.transform(train)
test_tokens = tokenizer.transform(test)

# Apply HashingTF and IDF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")

train_tf = hashingTF.transform(train_tokens)
idfModel = idf.fit(train_tf)
train_tfidf = idfModel.transform(train_tf)
test_tf = hashingTF.transform(test_tokens)
test_tfidf = idfModel.transform(test_tf)

# Train and evaluate logistic regression models
reg_param = 0.1
extract_prob = udf(lambda x: float(x[1]), FloatType())

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")

test_res = test.select('id')
for col in out_cols:
    lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=reg_param)
    lr_model = lr.fit(train_tfidf)
    
    train_res = lr_model.transform(train_tfidf)
    train_res = train_res.withColumn("proba", extract_prob("probability")).select("proba", "prediction", col)
    train_res = train_res.withColumnRenamed(col, "label")
    
    accuracy = evaluator.evaluate(train_res)
    print(f"Accuracy for label '{col}': {accuracy}")
    train_res.show(5)
    
    test_pred = lr_model.transform(test_tfidf)
    test_res = test_res.join(test_pred.select('id', 'probability'), on="id")
    test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")

# Save the test results
test_res.coalesce(1).write.csv('spark_lr_results.csv', mode='overwrite', header=True)

# Stop the SparkSession
spark.stop()