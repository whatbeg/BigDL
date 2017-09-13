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

from bigdl.dataset.transformer import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


def build_models(model_type='wide_n_deep', classNum=2):

    model = Sequential()
    wide_model = Concat(2)
    base_model = Sequential().add(Narrow(2, 1, 2006)).add(Reshape([2006]))
    crossed_model = Sequential().add(Narrow(2, 2007, 3000)).add(Reshape([3000]))
    deep_model = Sequential()
    deep_column = Concat(2)
    deep_column.add(Sequential().add(Narrow(2, 5007, 33)).add(Reshape([33])))
    deep_column.add(Sequential().add(Select(2, 5040)).add(LookupTable(1000, 8, 0.0)))   # workclass 100
    deep_column.add(Sequential().add(Select(2, 5041)).add(LookupTable(1000, 8, 0.0)))   # education 1000
    deep_column.add(Sequential().add(Narrow(2, 5042, 5)).add(Reshape([5])))
    deep_model.add(deep_column).add(Linear(54, 100)).add(ReLU()).add(Linear(100, 50)).add(ReLU())

    if model_type == 'wide_n_deep':
        wide_model.add(base_model)
        wide_model.add(crossed_model)
        wide_model.add(deep_model)
        model.add(wide_model).add(Linear(5056, classNum)).add(LogSoftMax())
        return model
    elif model_type == 'wide':
        wide_model.add(base_model)
        wide_model.add(crossed_model)
        model.add(wide_model).add(Linear(5006, classNum)).add(LogSoftMax())
        return model
    elif model_type == 'deep':
        model.add(deep_model).add(Linear(50, classNum)).add(LogSoftMax())
        return model
    else:
        raise ValueError("Not valid model type. Only for wide, deep, wide_n_deep!")


def get_data_rdd(sc, folder, data_type='train'):

    if data_type == 'train':
        data_tensor = folder + '/train_tensor.data'
    elif data_type == 'test':
        data_tensor = folder + '/test_tensor.data'
    else:
        raise ValueError("Not valid Data Type, only 'train' or 'test' !")

    features_label = sc.textFile(data_tensor)
    features = features_label.map(lambda x: x.split(',')[:-1])
    labels = features_label.map(lambda x: x.split(',')[-1]).map(lambda x: int(x))
    record = features.zip(labels).map(lambda features_label:
                                      Sample(features_label[0], features_label[1]+1, 3048, 1))
    return record


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="100")
    parser.add_option("-f", "--folder", dest="folder", default="")
    parser.add_option("-m", "--model", dest="model_type", default="wide_n_deep")
    parser.add_option("-l", "--lr", type=float, dest="learningRate", default="0.001")
    parser.add_option("-o", "--modelPath", dest="modelPath", default="/tmp/widedeep/20170913_164238/model.989")
    parser.add_option("-c", "--checkpointPath", dest="checkpointPath", default="/tmp/widedeep")
    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default="200")
    
    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="wide_n_deep", conf=create_spark_conf())
    init_engine()

    if options.action == "train":
        train_data = get_data_rdd(sc, options.folder, 'train')
        test_data = get_data_rdd(sc, options.folder, 'test')

        optimizer = Optimizer(
            model=build_models(options.model_type, 2),
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method=Adam(learningrate=options.learningRate, learningrate_decay=0.0005),
            end_trigger=MaxEpoch(options.maxEpoch),
            batch_size=options.batchSize
        )
        optimizer.set_validation(
            batch_size=options.batchSize,
            val_rdd=test_data,
            trigger=EveryEpoch(),
            val_method=[Top1Accuracy(), Loss()]
        )
        optimizer.set_checkpoint(EveryEpoch(), options.checkpointPath)
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()

        results = trained_model.test(test_data, options.batchSize, [Top1Accuracy(), Loss()])
        for result in results:
            print(result)
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        # Now modelPath format is changed, so test will failed in load.
        test_data = get_data_rdd(sc, options.folder, 'test')
        model = Model.load(options.modelPath)
        results = model.test(test_data, options.batchSize, [Top1Accuracy()])
        for result in results:
            print(result)
    sc.stop()
