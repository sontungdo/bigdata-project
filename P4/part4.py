import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder.appName("CensusIncomeClassification").getOrCreate()

# Load the data
# input_dir = "../../data-census/"
input_dir = sys.argv[1]
train_data = spark.read.csv(input_dir + "adult.data", header=False, inferSchema=True)
test_data = spark.read.csv(input_dir + "adult.test", header=False, inferSchema=True)

# Specify the column names
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]

train_data = train_data.toDF(*columns)
test_data = test_data.toDF(*columns)

# Preprocess the data
categorical_columns = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "sex", "native_country"]
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categorical_columns]

# Convert income to binary label
label_indexer = StringIndexer(inputCol="income", outputCol="label")

# Create a vector assembler
assembler = VectorAssembler(inputCols=[col+"_index" for col in categorical_columns] +
                            ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"],
                            outputCol="features")

# Create a pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers + [label_indexer, assembler])

# Fit the pipeline on the training data
train_data = pipeline.fit(train_data).transform(train_data)
test_data = pipeline.fit(test_data).transform(test_data)

# Create a random forest classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxBins=50)

# Train the random forest model
rf_model = rf.fit(train_data)

# Create a decision tree classifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxBins=50)

# Train the decision tree model
dt_model = dt.fit(train_data)

# Make predictions on the test data using random forest
rf_predictions = rf_model.transform(test_data)

# Make predictions on the test data using decision tree
dt_predictions = dt_model.transform(test_data)

# Evaluate the models
evaluator = BinaryClassificationEvaluator(labelCol="label")

rf_accuracy = evaluator.evaluate(rf_predictions)
dt_accuracy = evaluator.evaluate(dt_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)

# Stop the SparkSession
spark.stop()