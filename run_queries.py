import os
import time
import subprocess
import argparse

gap = "============================="
rule_jar_path = "/home/yjh/spark_tune/spark_ltr_qo/BeamSearchQueryExecutor/target/QueryExecutor.jar"
config_path = "/home/yjh/spark_tune/environment/spark-3.2.4/benchmarks/conf"

sql_path = "/home/yjh/spark_tune/environment/spark-3.2.4/benchmarks/sqls/stack"
database = "stack"

# Resource allocation
DEFAULT_KNOBS = {
    'spark.driver.cores': 6,
    'spark.driver.memory': '10240m',
    'spark.executor.instances': 6,
    'spark.executor.cores': 6,
    'spark.executor.memory': '10240m'
}

EXTRA_KNOBS = {
    'spark.master': 'yarn',
    'spark.submit.deployMode': 'client',
    'spark.eventLog.enabled': 'true',
    'spark.eventLog.compress': 'false',
    'spark.yarn.jars': 'hdfs://node183:9000/home/yjh/spark_tune/environment/spark-3.3.0/jars/*.jar',
    'spark.eventLog.dir': 'hdfs://node183:9000/home/yjh/spark_tune/environment/spark-3.2.4/log',
    'spark.yarn.maxAppAttempts': 1,
    'spark.sql.catalogImplementation': 'hive',
    'spark.sql.adaptive.optimizer.excludedRules': 'org.apache.spark.sql.execution.adaptive.AQEPropagateEmptyRelation',
    'spark.sql.cbo.enabled': 'true',
    'spark.sql.cbo.joinReorder.dp.threshold': '18',
    'spark.card.rpc.port': 9009,
    'spark.card.rpc.host': 'localhost',
    'spark.card.beamsearch.k': '4',
    'spark.sql.extensions': 'org.example.CustomExtension',  # 不要改这个
    'spark.card.join.reorder.class': 'SplitOrLeftDeepJoinReorder'
}


def gen_sql_list():
    all_sql_list = []
    for root, dirs, files in os.walk(sql_path):
        for file in files:
            all_sql_list.append(file[:-4])
    return all_sql_list


def write_config_file(config, file_name):
    with open(file_name, "w") as conf_file:
        for knob, value in EXTRA_KNOBS.items():
            config[knob] = value
        for conf in config:
            conf_file.write(f"{conf} {config[conf]}\n")


def run_task(task_id, config):
    cur_time = int(round(time.time() * 1000))
    config_file_path = f"{config_path}/{cur_time}.conf"
    write_config_file(config, config_file_path)

    cmd = f"spark-sql --database {database} -f {sql_path}/{task_id}.sql --name {task_id}_{name_suffix} --properties-file {config_file_path} --jars {rule_jar_path}"
    print(gap, flush=True)
    print(cmd, flush=True)
    try:
        subprocess.run(cmd, timeout=300, shell=True)
    except subprocess.TimeoutExpired:
        print("Time taken: 300.00 seconds")


def init_default_data():
    sqls = gen_sql_list()
    for sql in sqls:
        run_task(sql, DEFAULT_KNOBS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_suffix', type=str, default='stack_lero_800')
    parser.add_argument('--dataset', type=str, default='imdb_10x')
    opt = parser.parse_args()
    name_suffix = opt.name_suffix
    dataset = opt.dataset
    if dataset == 'tpch':
        sql_path = "/home/yjh/spark_tune/environment/spark-3.2.4/benchmarks/sqls/tpch-spj"
        database = "tpch_100"
    elif dataset == 'stack':
        sql_path = "/home/yjh/spark_tune/environment/spark-3.2.4/benchmarks/sqls/stack"
        database = "stack"
    elif dataset == 'imdb_10x':
        sql_path = "/home/yjh/spark_tune/environment/spark-3.2.4/benchmarks/sqls/imdb"
        database = "imdb_10x"

    init_default_data()
