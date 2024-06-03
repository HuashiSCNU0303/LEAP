# LEAP

This is the source code of paper "LEAP: A Low-cost Spark SQL Query Optimizer using Pairwise Comparison"

## Run steps

1. Run the following command to build the modules,
```shell
cd ./BeamSearchQueryExecutor
mvn clean generate-sources package
```

2. Update the corresponding parameters in ./leap_server/config.py and ./run_queries.py

3. Run the following command to start the LEAP server, and execute the queries,
```shell
cd ../
./start_server.sh --dataset imdb_10x
```
