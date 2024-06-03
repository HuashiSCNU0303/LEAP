#!/bin/bash

dataset="stack"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

python -u ./leap_server/card_est_server.py --dataset_name ${dataset} > ./leap_server/server_${dataset}.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID

python -u run_queries.py --dataset ${dataset} --name_suffix ${dataset}_ours > ${dataset}_ours.log 2>&1

kill $SERVER_PID
echo "The server has terminated..."

sleep 5
