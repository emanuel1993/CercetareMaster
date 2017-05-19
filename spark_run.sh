#!/bin/bash

scp -pr /home/student/Desktop/SparkEmanuel hadoop-master:/home/hadoop/
ssh hadoop-master "/opt/spark/bin/spark-submit --master spark://172.19.3.36:7077 /home/hadoop/SparkEmanuel/main_distrib.py && cat /home/hadoop/resultEmanuel.txt"