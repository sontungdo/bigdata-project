#!/bin/bash
source ../../env.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /data-census/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /data-census/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data-census/adult.data /data-census/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data-census/adult.test /data-census/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 part3.py hdfs://$SPARK_MASTER:9000/data-census/input/
