#!/bin/bash
#/opt/spark/sbin/stop-all.sh
#/opt/hadoop/sbin/stop-all.sh
for i in `seq 1 10`
do
	#/opt/hadoop/sbin/start-all.sh
	#/opt/spark/sbin/start-all.sh
	echo 'Execution '$i

	# /opt/spark/bin/spark-submit --master spark://172.19.3.36:7077 /home/hadoop/CercetareMaster/main_distrib.py
	/opt/spark/bin/spark-submit --master yarn --deploy-mode client /home/hadoop/CercetareMaster/main_distrib.py

    sleep 20

	# stop Hadoop and Spark
	#/opt/spark/sbin/stop-all.sh
	#/opt/hadoop/sbin/stop-all.sh
done
