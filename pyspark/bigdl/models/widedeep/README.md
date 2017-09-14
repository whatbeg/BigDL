# Wide and Deep Learning Model on Census

Wide and Deep Learning Model, proposed by Google, is a DNN and Linear mixed model.
Wide and Deep Learning has been used for Google App Store for their recommendation.

For detail information, please refer to:
[TensorFlow Tutorial](https://www.tensorflow.org/tutorials/wide_and_deep)
[Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)

Census Income dataset was extracted by Barry Becker from the 1994 Census database.
Prediction task is to determine whether a person makes over 50K a year based on Census data.
Please refer to <https://archive.ics.uci.edu/ml/datasets/census+income>

## How to run this example:

Please download the Census data in advance, and put them into a folder, e.g. /tmp/census

```
/tmp/census tree .
.
├── adult.data
├── adult.test

```

And then run `preprocessing.py` to get preprocessed data `train.data` and `test.data`.

```
/tmp/census tree .
.
├── adult.data
├── adult.test
├── train.data
├── test.data
```

We would train a Wide and Deep model in spark local mode with the following commands and you can distribute it across cluster by modifying the spark master and the executor cores.

```
    BigDL_HOME=...
    PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.2.0-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH

    MASTER=local[*]

    logname=`date +%m%d%H%M`

    spark-submit \
        --master ${MASTER} \
        --driver-cores 20 \
        --driver-memory 180g \
        --executor-cores 20  \
        --executor-memory 180g \
        --total-executor-cores 20 \
        --conf spark.rpc.message.maxSize=1024 \
        --py-files ${PYTHON_API_PATH},${BigDL_HOME}/pyspark/bigdl/models/widedeep/widedeep.py  \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-0.2.0-jar-with-dependencies.jar \
        ${BigDL_HOME}/pyspark/bigdl/models/widedeep/widedeep.py \
        --action train \
        --folder path/to/census \
        --batchSize 1280 \
        --maxEpoch 100 \
        --lr 0.001 \
        --model wide_n_deep |& tee WD${logname}.log

```


* ```--action``` it can be train or test.

* ```--folder``` option can be used to set data folder, which contains preprocessed data.

* ```--batchSize``` option can be used to set batch size, the default value is 128.

* ```--maxEpoch``` option can be used to control how to end the training process.

* ```--modelPath``` option can be used to set model path for testing, the default value is /tmp/widedeep/2017mmdd_HHMMSS/model.989.

* ```--checkpointPath``` option can be used to set checkpoint path for saving model, the default value is /tmp/widedeep/.

To verify the accuracy, search "accuracy" from log:

```
INFO  DistriOptimizer$:247 - [Epoch 1 0/32561][Iteration 1][Wall Clock 0.0s] Train 1280 in xx seconds. Throughput is xx records/second.

INFO  DistriOptimizer$:629 - Top1Accuracy is Accuracy(correct: 13843, count: 16281, accuracy: 0.8502548983477674)

```