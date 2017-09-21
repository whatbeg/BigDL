/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.widedeep_tutorial

import com.intel.analytics.bigdl.dataset.SparseTensorMiniBatch
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Train {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Wide and Deep Learning on Census")
        // Will throw exception without this config when has only one executor
        .set("spark.rpc.message.maxSize", "1024")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = param.batchSize

      val trainData = param.folder + "/train.data"
      val testData = param.folder + "/test.data"

      val trainDataSet = load(sc, trainData, "Train")
      val validateSet = load(sc, testData, "Test")

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        SparseWideDeep[Float](modelType = "wide_n_deep", classNum = 2)
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new Adam[Float](
          learningRate = param.learningRate,
          learningRateDecay = param.learningRateDecay
        )
      }

      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainDataSet,
        criterion = ClassNLLCriterion[Float](),
        batchSize = batchSize,
        miniBatch = new SparseTensorMiniBatch[Float](Array(
          Tensor.sparse(Array(5006), 1),
          Tensor(1, 40)),
          Array(Tensor(1, 1)))
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      val logdir = "widedeep"
      val appName = s"${sc.applicationId}"
      val trainSummary = TrainSummary(logdir, appName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
      val validationSummary = ValidationSummary(logdir, appName)

      optimizer
        .setOptimMethod(optimMethod)
        .setTrainSummary(trainSummary)
        .setValidationSummary(validationSummary)
        .setValidation(Trigger.everyEpoch,
          validateSet, Array(new Top1Accuracy[Float],
            new Loss[Float](ClassNLLCriterion[Float]())),
          batchSize = batchSize,
          miniBatch = new SparseTensorMiniBatch[Float](Array(
            Tensor.sparse(Array(5006), 1),
            Tensor(1, 40)),
            Array(Tensor(1, 1))))
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
      sc.stop()
    })
  }
}
