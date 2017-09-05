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

package com.intel.analytics.bigdl.models.widedeep

import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dataset.SparseTensorMiniBatch
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

object Train {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.DEBUG)

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
        WideDeepWithSparse[Float](modelType = "wide_n_deep", classNum = 2)
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
        criterion = new CrossEntropyCriterion[Float](),
        batchSize = batchSize,
        miniBatch = new SparseTensorMiniBatch[Float](Array(
          Tensor.sparse(Array(1023213), 1),
          Tensor(1, 11)),
          Array(Tensor(1, 1)))
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      optimizer
        .setOptimMethod(optimMethod)
        .setValidation(Trigger.everyEpoch,
          validateSet, Array(new Top1Accuracy[Float], new Loss[Float]),
          batchSize = batchSize,
          miniBatch = new SparseTensorMiniBatch[Float](Array(
            Tensor.sparse(Array(1023213), 1),
            Tensor(1, 11)),
            Array(Tensor(1, 1))))
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
      sc.stop()
    })
  }
}
