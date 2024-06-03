#!/bin/bash

dataset="stack"

python -u card_est_server.py > server_${dataset}.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID

python -u /home/yjh/spark_tune/spark_ltr_qo_clean/run_queries.py --dataset ${dataset} --name_suffix ${dataset}_ours > /home/yjh/spark_tune/spark_ltr_qo_clean/${dataset}_ours.log 2>&1

kill $SERVER_PID
echo "服务器已停止。"

sleep 5
