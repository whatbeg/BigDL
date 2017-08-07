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

package com.intel.analytics.bigdl.models

import java.io.File

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.{DenseTensor, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.models.widedeep.Utils
import org.apache.spark.{SparkConf, SparkContext}


class TestSpec extends FlatSpec with BeforeAndAfter with Matchers {
  var sc: SparkContext = null
  val nodeNumber = 1
  val coreNumber = 1

  before {
    Engine.init(nodeNumber, coreNumber, true)
    val conf = new SparkConf().setMaster("local[1]").setAppName("DataSetSpec")
    sc = new SparkContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  val resource = getClass().getClassLoader().getResource("wide_deep")
  println(resource)

  val s = Utils

  val dataSet = com.intel.analytics.bigdl.models.widedeep.Utils.loadTrain(
    processPath(resource.getPath()) + File.separator + "train.data")
  print(dataSet.length)
  // println(Tensor.dense(dataSet(0)))

}
