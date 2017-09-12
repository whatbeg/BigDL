#!/usr/bin/env bash

BigDL_HOME=/root/BigDL
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH

# MASTER=spark://172.168.2.160:7077
MASTER=local[4]

PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH

logname=`date +%m%d%H%M`

spark-submit \
    --master ${MASTER} \
    --driver-cores 4 \
    --driver-memory 40g \
    --executor-cores 4  \
    --executor-memory 80g \
    --total-executor-cores 4 \
    --conf spark.rpc.message.maxSize=1024 \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar \
    --class com.intel.analytics.bigdl.models.widedeep_tutorial.Train \
    ${BigDL_JAR_PATH} \
    -f ${BigDL_HOME}/census \
    -b 1152 \
    -e 50 \
    -r 0.001 |& tee LOG/BigDL_3k_sparse_1152_local24_TEST${logname}.log
