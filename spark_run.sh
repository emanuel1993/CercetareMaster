#!/bin/bash

scp -pr /home/student/Desktop/EmanuelCercetare hadoop-master:/home/hadoop/
ssh hadoop-master "/opt/spark/bin/spark-submit --master yarn --deploy-mode client /home/hadoop/EmanuelCercetare/main_distrib.py && cat /home/hadoop/resultEmanuel.txt && rm /home/hadoop/resultEmanuel.txt"