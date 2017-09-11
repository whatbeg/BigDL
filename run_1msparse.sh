#!/bin/bash

BigDL_HOME=/root/BigDL
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH

# MASTER=spark://172.168.2.160:7077
MASTER=local[20]

PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH

logname=`date +%m%d%H%M`

spark-submit \
    --master ${MASTER} \
    --driver-cores 20 \
    --driver-memory 180g \
    --executor-cores 20  \
    --executor-memory 180g \
    --total-executor-cores 20 \
    --conf spark.rpc.message.maxSize=200 \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar \
    --class com.intel.analytics.bigdl.models.widedeep.Train \
    ${BigDL_JAR_PATH} \
    -f hdfs://172.168.2.160:9000/census \
    -b 1280 \
    -e 2 \
    -r 0.01 |& tee LOG/wd${logname}.log

# grep -E 'DistriOptimizer\$|Test result' LOG/wd${logname}.log > LOG/wd${logname}_F.log
# grep -E 'Top1Accuracy|Test result' LOG/wd${logname}_F.log > LOG/wd${logname}_SIM.log
