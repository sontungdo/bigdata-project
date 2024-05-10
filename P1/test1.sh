#!/bin/bash
source ../../env.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /toxic-comment/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /toxic-comment/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data-comment/train.csv /toxic-comment/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data-comment/test.csv /toxic-comment/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 part1.py hdfs://$SPARK_MASTER:9000/toxic-comment/input/
