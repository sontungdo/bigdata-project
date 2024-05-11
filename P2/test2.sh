#!/bin/bash
source ../../env.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /heart-disease/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /heart-disease/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/train.csv /heart-disease/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 part2.py hdfs://$SPARK_MASTER:9000/heart-disease/input/
