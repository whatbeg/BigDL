#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Still in experimental stage!

from optparse import OptionParser
import sys
import pandas as pd
import numpy as np
# from ...dataset.transformer import *
# from ...nn.layer import *
# from ...nn.criterion import *
# from ...optim.optimizer import *
# from ...util.common import *

from dataset.transformer import *
from nn.layer import *
from nn.criterion import *
from optim.optimizer import *
from util.common import *

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def build_models(classNum):

    model = Sequential()
    submodel = Concat(2)
    submodel.add(Sequential().add(Select(2, 1)).add(Reshape([1])))
    submodel.add(Sequential().add(Select(2, 2)).add(Reshape([1])))
    submodel.add(Sequential().add(Select(2, 3)).add(Reshape([1])))
    submodel.add(Sequential().add(Select(2, 4)).add(Reshape([1])))
    submodel.add(Sequential().add(Select(2, 5)).add(Reshape([1])))
    submodel.add(Sequential().add(Select(2, 6)).add(Reshape([1])))
    submodel.add(Sequential().add(Select(2, 7)).add(Reshape([1])))
    deep_model = Sequential()
    deep_column = Concat(2)
    deep_column.add(Sequential().add(Select(2, 8)).add(LookupTable(9, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 9)).add(LookupTable(16, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 10)).add(LookupTable(2, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 11)).add(LookupTable(6, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 12)).add(LookupTable(42, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 13)).add(LookupTable(15, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 14)).add(Reshape([1])))
    deep_column.add(Sequential().add(Select(2, 15)).add(Reshape([1])))
    deep_column.add(Sequential().add(Select(2, 16)).add(Reshape([1])))
    deep_column.add(Sequential().add(Select(2, 17)).add(Reshape([1])))
    deep_column.add(Sequential().add(Select(2, 18)).add(Reshape([1])))
    deep_model.add(deep_column).add(Linear(53, 100)).add(ReLU()).add(Linear(100, 50)).add(ReLU())
    submodel.add(deep_model)
    model.add(submodel).add(Linear(57, classNum)).add(LogSoftMax())
    return model


def get_data_rdd(sc, data_type='train'):

    if data_type == 'train':
        data_tensor = './census/train_tensor.data'
        data_label = './census/train_label.data'
    elif data_type == 'test':
        data_tensor = './census/test_tensor.data'
        data_label = './census/test_label.data'
    else:
        raise ValueError("Not valid Data Type, only 'train' or 'test' !")
    features = np.loadtxt(data_tensor, delimiter=',')
    labels = np.loadtxt(data_label)
    features = sc.parallelize(features)
    labels = sc.parallelize(labels)
    record = features.zip(labels).map(lambda features_label:
                                      Sample.from_ndarray(features_label[0], features_label[1]+1))
    # record.collect()
    return record


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")

    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="wide_n_deep", conf=create_spark_conf())
    init_engine()

    if options.action == "train":
        train_data = get_data_rdd(sc, 'train')
        test_data = get_data_rdd(sc, 'test')
        state = {"learningRate": 0.01}
        optimizer = Optimizer(
            model=build_models(2),
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method="Adam",
            state=state,
            end_trigger=MaxEpoch(20),
            batch_size=int(options.batchSize))
        optimizer.set_validation(
            batch_size=32,
            val_rdd=test_data,
            trigger=EveryEpoch(),
            val_method=["Top1Accuracy"]
        )
        optimizer.set_checkpoint(EveryEpoch(), "/tmp/wide_deep")
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        test_data = get_data_rdd(sc, 'test')
        # TODO: Pass model path through external parameter
        model = Model.from_path("/tmp/wide_deep/wide_deep-model.470")
        results = model.test(test_data, 32, ["Top1Accuracy"])
        for result in results:
            print(result)
