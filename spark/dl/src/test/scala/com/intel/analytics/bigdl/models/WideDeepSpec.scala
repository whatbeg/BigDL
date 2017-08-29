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

import com.intel.analytics.bigdl.models.widedeep.{WideDeep, WideDeepWithSparse}
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File, RandomGenerator, T}

import scala.util.Random

class WideDeepSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "wide deep" should "forward" in {
    Random.setSeed(100)
    RandomGenerator.RNG.setSeed(100)
    val model = WideDeep("wide_n_deep", 2)
    val sparseModel = WideDeepWithSparse("wide_n_deep", 2)
    val wideColumn = Tensor(4, 1023213)
    val deepColumn = Tensor(4, 11)
    for (i <- 1 to 4) {
      wideColumn.setValue(i, Random.nextInt(2) + 1, Random.nextInt(2))
      wideColumn.setValue(i, Random.nextInt(1000) + 3, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 1003, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 2003, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(100) + 3003, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(100) + 3103, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(11) + 3203, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(10000) + 3214, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(10000) + 13214, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000000) + 23214, Random.nextInt(10))
      deepColumn.setValue(i, 1, 1 + Random.nextInt(100))
      deepColumn.setValue(i, 2, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, 3, 1 + Random.nextInt(2))
      deepColumn.setValue(i, 4, 1 + Random.nextInt(100))
      deepColumn.setValue(i, 5, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, 6, 1 + Random.nextInt(1000))
      deepColumn.narrow(2, 7, 5).apply1(_ => Random.nextInt(3))
    }
    val wdColumn = Tensor(4, 1023224)
    wdColumn.narrow(2, 1, 1023213).copy(wideColumn)
    wdColumn.narrow(2, 1023214, 11).copy(deepColumn)
    val output = model.forward(wdColumn)
    val sparseInput = T(Tensor.sparse(wideColumn), deepColumn)
    sparseModel.getParameters()._1.copy(model.getParameters()._1)
    val sparseOutput = sparseModel.forward(sparseInput)

    output shouldEqual sparseOutput
  }

  "wide deep" should "forward & backward" in {
    Random.setSeed(100)
    RandomGenerator.RNG.setSeed(100)
    val model = WideDeep("wide_n_deep", 2)
    val sparseModel = WideDeepWithSparse("wide_n_deep", 2)
      // create a wide&deep like input
    val wideColumn = Tensor(4, 1023213)
    val deepColumn = Tensor(4, 11)
    for (i <- 1 to 4) {
      wideColumn.setValue(i, Random.nextInt(2) + 1, Random.nextInt(2))
      wideColumn.setValue(i, Random.nextInt(1000) + 3, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 1003, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 2003, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(100) + 3003, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(100) + 3103, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(11) + 3203, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(10000) + 3214, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(10000) + 13214, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000000) + 23214, Random.nextInt(10))
      deepColumn.setValue(i, 1, 1 + Random.nextInt(100))
      deepColumn.setValue(i, 2, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, 3, 1 + Random.nextInt(2))
      deepColumn.setValue(i, 4, 1 + Random.nextInt(100))
      deepColumn.setValue(i, 5, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, 6, 1 + Random.nextInt(1000))
      deepColumn.narrow(2, 7, 5).apply1(_ => Random.nextInt(3))
    }
    val wdColumn = Tensor(4, 1023224)
    wdColumn.narrow(2, 1, 1023213).copy(wideColumn)
    wdColumn.narrow(2, 1023214, 11).copy(deepColumn)
    val output = model.forward(wdColumn).toTensor[Float]
    val sparseInput = T(Tensor.sparse(wideColumn), deepColumn)
    sparseModel.getParameters()._1.copy(model.getParameters()._1)
    sparseModel.forward(sparseInput)
    val sparseOutput = sparseModel.forward(sparseInput)

    val gradOutput = output.clone().rand()
    val denseGradInput = model.backward(wdColumn, gradOutput)
    sparseModel.backward(sparseInput, gradOutput)
    sparseModel.zeroGradParameters()
    val sparseGradInput = sparseModel.backward(sparseInput, gradOutput)

    val a = sparseModel.getParametersTable()
    val b = model.getParametersTable()

    sparseModel.getParameters()._2.equals(model.getParameters()._2) shouldEqual true
    // parameters of sparse model should equal to parameters of dense model

    output shouldEqual sparseOutput  // dense output should equal to sparse output
  }

}
