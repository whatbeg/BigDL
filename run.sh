#!/usr/bin/env bash

BigDL_HOME=/home/huqiu/BigDL
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH

source $BigDL_HOME/dist/bin/bigdl.sh

MASTER=local[4]
PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH

logname=`date +%m%d%H%M`

spark-submit \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 2g \
    --total-executor-cores 2 \
    --executor-cores 2  \
    --executor-memory 4g \
    --conf spark.akka.frameSize=64 \
    --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/dl/models/widedeeprecommender/widedeeprecommender.py  \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/dl/models/widedeeprecommender/widedeeprecommender.py \
    --action train |& tee LOG/${logname}.log

grep 'DistriOptimizer\$' LOG/${logname}.log > LOG/${logname}_F.log
grep 'Top1Accuracy' LOG/${logname}_F.log > LOG/${logname}_SIMPLE.log

