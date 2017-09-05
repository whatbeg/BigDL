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

import com.intel.analytics.bigdl.dataset.TensorSample
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.{DenseTensor, SparseTensor, Storage, Tensor}
import com.intel.analytics.bigdl.models.widedeep.{WideDeep, WideDeepWithSparse}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.{SparkConf, SparkContext}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.models.widedeep.Utils._
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion


class WideDeepUtilSpec extends FlatSpec with BeforeAndAfter with Matchers {

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }


  "foward" should "get through" in {
    var sc: SparkContext = null
    val nodeNumber = 1
    val coreNumber = 1

    Engine.init(nodeNumber, coreNumber, true)
    val conf = new SparkConf().setMaster("local[1]").setAppName("WideDeepUtilSpec")
    sc = new SparkContext(conf)

    val resource = getClass().getClassLoader().getResource("wide_deep")

    val dataSet = com.intel.analytics.bigdl.models.widedeep.Utils.load2(sc,
      processPath(resource.getPath()) + File.separator + "train.data", "Train")

    val input = dataSet.take(2)
    val sparseModel = WideDeepWithSparse[Float]("wide_n_deep", 2)
    println(input)
    println(input.size)
    println(input(0)(1))
    val sps_result = Tensor.sparse[Float](Array(1023213), 15)
    sps_result.concat(1, T(input(0)(0), input(1)(0)), sps_result)
    println(sps_result)
    val den_result = Tensor[Float](1, 11)
    den_result.concat(1, T(input(0)(1), input(1)(1)), den_result)
    println(den_result)
    val lbl = Tensor[Float](1, 1)
    lbl.concat(1, T(input(0)(2), input(1)(2)), lbl)
    println(lbl)
    val sparseOutput = sparseModel.forward(T(sps_result, den_result))
    val criterion = new CrossEntropyCriterion[Float]()
    println(sparseOutput.toTensor[Float])
    val loss = criterion.forward(sparseOutput.toTensor[Float], lbl)
    println(loss)
    sc.stop()
  }

}
