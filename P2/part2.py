import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when, lit
import matplotlib.pyplot as plt

# Create a SparkSession
spark = SparkSession.builder \
    .appName("HeartDiseasePredictor") \
    .getOrCreate()

# Read the Framingham Heart Study dataset
# data = spark.read.csv("../../data/framingham.csv", header=True, inferSchema=True)
data = spark.read.csv(sys.argv[1] + "framingham.csv", header=True, inferSchema=True)

# Perform data preprocessing
data = data.drop("education")
data = data.withColumnRenamed("male", "Sex_male")

# Convert data types to numeric
data = data.withColumn("cigsPerDay", col("cigsPerDay").cast("double"))
data = data.withColumn("totChol", col("totChol").cast("double"))
data = data.withColumn("glucose", col("glucose").cast("double"))

# Drop rows with null values
data = data.dropna()

# Select the relevant features
features = ["age", "Sex_male", "cigsPerDay", "totChol", "sysBP", "glucose"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(data)

# Split the data into training and testing sets
(trainingData, testData) = data.randomSplit([0.8, 0.2], seed=5)

# Train the logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="TenYearCHD")
model = lr.fit(trainingData)

# Make predictions on the testing set
predictions = model.transform(testData)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="TenYearCHD")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# Calculate performance metrics
tp = predictions.filter((col("prediction") == 1) & (col("TenYearCHD") == 1)).count()
tn = predictions.filter((col("prediction") == 0) & (col("TenYearCHD") == 0)).count()
fp = predictions.filter((col("prediction") == 1) & (col("TenYearCHD") == 0)).count()
fn = predictions.filter((col("prediction") == 0) & (col("TenYearCHD") == 1)).count()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Plot the ROC curve and calculate AUC score
# roc = model.summary.roc.toPandas()
# plt.plot(roc["FPR"], roc["TPR"])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.show()

auc = evaluator.evaluate(predictions)
print("AUC Score:", auc)