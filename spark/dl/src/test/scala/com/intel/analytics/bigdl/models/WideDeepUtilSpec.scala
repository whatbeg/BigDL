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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.{SparkConf, SparkContext}
import com.intel.analytics.bigdl.models.widedeep_tutorial.SparseWideDeep
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion


class WideDeepUtilSpec extends FlatSpec with BeforeAndAfter with Matchers {

  var sc: SparkContext = null
  val nodeNumber = 1
  val coreNumber = 1

  Engine.init(nodeNumber, coreNumber, true)
  val conf = new SparkConf().setMaster("local[1]").setAppName("WideDeepUtilSpec")
  sc = new SparkContext(conf)

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }


  "forward" should "get through" in {

    val resource = getClass().getClassLoader().getResource("wide_deep")

    val dataSet = com.intel.analytics.bigdl.models.widedeep_tutorial.Utils.load2(sc,
      processPath(resource.getPath()) + File.separator + "train.data", "Train")

    val input = dataSet.take(4)
    val sparseModel = SparseWideDeep[Float]("wide_n_deep", 2)
    val sps_result = Tensor.sparse[Float](Array(5006), 3)
    sps_result.concat(1, T(input(0)(0), input(1)(0),
      input(2)(0), input(3)(0)), sps_result)

    sps_result.nElement() shouldEqual 44

    val den_result = Tensor[Float](1, 40)
    den_result.concat(1, T(input(0)(1), input(1)(1),
      input(2)(1), input(3)(1)), den_result)

    den_result.size() shouldEqual (4, 40)

    val lbl = Tensor[Float](1, 1)
    lbl.concat(1, T(input(0)(2), input(1)(2), input(2)(2), input(3)(2)), lbl)

    lbl.size() shouldEqual (4, 1)

    val sparseOutput = sparseModel.forward(T(sps_result, den_result))

    sparseOutput.toTensor[Float].size() shouldEqual (4, 2)

    val criterion = new CrossEntropyCriterion[Float]()
    criterion.forward(sparseOutput.toTensor[Float], lbl)
  }

  sc.stop()

}
