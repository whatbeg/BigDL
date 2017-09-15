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

import com.intel.analytics.bigdl.models.widedeep_tutorial.{WideDeep, SparseWideDeep}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}

import scala.util.Random

class WideDeepSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "wide" should "forward" in {
    Random.setSeed(100)
    RandomGenerator.RNG.setSeed(100)
    val model = WideDeep("wide", 2)
    val sparseModel = SparseWideDeep("wide", 2)
    val wideColumn = Tensor(4, 5006)
    for (i <- 1 to 4) {
      wideColumn.setValue(i, 1, Random.nextInt(10))
      wideColumn.setValue(i, 2, Random.nextInt(10))
      wideColumn.setValue(i, 3, Random.nextInt(10))
      wideColumn.setValue(i, 4, Random.nextInt(10))
      wideColumn.setValue(i, 5, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 6, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 1006, Random.nextInt(10))
      wideColumn.setValue(i, 2006, Random.nextInt(11))
      wideColumn.setValue(i, Random.nextInt(1000) + 2007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 3007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 4007, Random.nextInt(10))
    }

    val output = model.forward(wideColumn)
    val wdColumn = wideColumn.narrow(2, 1, 1000)
    val sparseInput = Tensor.sparse(wideColumn)
    sparseModel.getParameters()._1.copy(model.getParameters()._1)
    val sparseOutput = sparseModel.forward(sparseInput)

    output shouldEqual sparseOutput
  }

  "deep" should "forward" in {
    Random.setSeed(100)
    RandomGenerator.RNG.setSeed(100)
    val model = WideDeep("deep", 2)
    val sparseModel = SparseWideDeep("deep", 2)
    val wideColumn = Tensor(4, 5006)
    val deepColumn = Tensor(4, 40)
    for (i <- 1 to 4) {
      wideColumn.setValue(i, 1, Random.nextInt(10))
      wideColumn.setValue(i, 2, Random.nextInt(10))
      wideColumn.setValue(i, 3, Random.nextInt(10))
      wideColumn.setValue(i, 4, Random.nextInt(10))
      wideColumn.setValue(i, 5, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 6, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 1006, Random.nextInt(10))
      wideColumn.setValue(i, 2006, Random.nextInt(11))
      wideColumn.setValue(i, Random.nextInt(1000) + 2007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 3007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 4007, Random.nextInt(10))

      deepColumn.setValue(i, Random.nextInt(9) + 1, 1 + Random.nextInt(100))
      deepColumn.setValue(i, Random.nextInt(16) + 10, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, Random.nextInt(2) + 26, 1 + Random.nextInt(2))
      deepColumn.setValue(i, Random.nextInt(6) + 28, 1 + Random.nextInt(100))
      deepColumn.setValue(i, 34, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, 35, 1 + Random.nextInt(1000))
      deepColumn.narrow(2, 36, 5).apply1(_ => Random.nextInt(100))
    }
    val wdColumn = Tensor(4, 5046)
    wdColumn.narrow(2, 1, 5006).copy(wideColumn)
    wdColumn.narrow(2, 5007, 40).copy(deepColumn)

    val output = model.forward(wdColumn)
    sparseModel.getParameters()._1.copy(model.getParameters()._1)
    val sparseOutput = sparseModel.forward(deepColumn)

    output shouldEqual sparseOutput
  }

  "wide deep" should "forward" in {
    Random.setSeed(100)
    RandomGenerator.RNG.setSeed(100)
    val model = WideDeep("wide_n_deep", 2)
    val sparseModel = SparseWideDeep("wide_n_deep", 2)
    val wideColumn = Tensor(4, 5006)
    val deepColumn = Tensor(4, 40)
    for (i <- 1 to 4) {
      wideColumn.setValue(i, 1, Random.nextInt(10))
      wideColumn.setValue(i, 2, Random.nextInt(10))
      wideColumn.setValue(i, 3, Random.nextInt(10))
      wideColumn.setValue(i, 4, Random.nextInt(10))
      wideColumn.setValue(i, 5, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 6, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 1006, Random.nextInt(10))
      wideColumn.setValue(i, 2006, Random.nextInt(11))
      wideColumn.setValue(i, Random.nextInt(1000) + 2007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 3007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 4007, Random.nextInt(10))

      deepColumn.setValue(i, Random.nextInt(9) + 1, 1 + Random.nextInt(100))
      deepColumn.setValue(i, Random.nextInt(16) + 10, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, Random.nextInt(2) + 26, 1 + Random.nextInt(2))
      deepColumn.setValue(i, Random.nextInt(6) + 28, 1 + Random.nextInt(100))
      deepColumn.setValue(i, 34, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, 35, 1 + Random.nextInt(1000))
      deepColumn.narrow(2, 36, 5).apply1(_ => Random.nextInt(100))
    }
    val wdColumn = Tensor(4, 5046)
    wdColumn.narrow(2, 1, 5006).copy(wideColumn)
    wdColumn.narrow(2, 5007, 40).copy(deepColumn)
    val output = model.forward(wdColumn)
    val sparseInput = T(Tensor.sparse(wideColumn), deepColumn)
    sparseModel.getParameters()._1.copy(model.getParameters()._1)
    val sparseOutput = sparseModel.forward(sparseInput)
    val sparseOutput2 = sparseModel.forward(sparseInput)

    output shouldEqual sparseOutput
  }

  "wide deep" should "forward & backward" in {
    Random.setSeed(100)
    RandomGenerator.RNG.setSeed(100)
    val model = WideDeep("wide_n_deep", 2)
    val sparseModel = SparseWideDeep("wide_n_deep", 2)
    // create a wide&deep like input
    val wideColumn = Tensor(4, 5006)
    val deepColumn = Tensor(4, 40)
    for (i <- 1 to 4) {
      wideColumn.setValue(i, 1, Random.nextInt(10))
      wideColumn.setValue(i, 2, Random.nextInt(10))
      wideColumn.setValue(i, 3, Random.nextInt(10))
      wideColumn.setValue(i, 4, Random.nextInt(10))
      wideColumn.setValue(i, 5, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 6, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 1006, Random.nextInt(10))
      wideColumn.setValue(i, 2006, Random.nextInt(11))
      wideColumn.setValue(i, Random.nextInt(1000) + 2007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 3007, Random.nextInt(10))
      wideColumn.setValue(i, Random.nextInt(1000) + 4007, Random.nextInt(10))

      deepColumn.setValue(i, Random.nextInt(9) + 1, 1 + Random.nextInt(100))
      deepColumn.setValue(i, Random.nextInt(16) + 10, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, Random.nextInt(2) + 26, 1 + Random.nextInt(2))
      deepColumn.setValue(i, Random.nextInt(6) + 28, 1 + Random.nextInt(100))
      deepColumn.setValue(i, 34, 1 + Random.nextInt(1000))
      deepColumn.setValue(i, 35, 1 + Random.nextInt(1000))
      deepColumn.narrow(2, 36, 5).apply1(_ => Random.nextInt(100))
    }
    val wdColumn = Tensor(4, 5046)
    wdColumn.narrow(2, 1, 5006).copy(wideColumn)
    wdColumn.narrow(2, 5007, 40).copy(deepColumn)
    val output = model.forward(wdColumn).toTensor[Float]
    val sparseInput = T(Tensor.sparse(wideColumn), deepColumn)
    // copy dense model weight to sparse
    sparseModel.getParameters()._1.copy(model.getParameters()._1)
    sparseModel.forward(sparseInput)
    val sparseOutput = sparseModel.forward(sparseInput)

    val gradOutput = output.clone().rand()
    val denseGradInput = model.backward(wdColumn, gradOutput)
    sparseModel.backward(sparseInput, gradOutput)
    sparseModel.zeroGradParameters()
    val sparseGradInput = sparseModel.backward(sparseInput, gradOutput)
    val sgi2 = sparseGradInput.toTable[Tensor[Float]](2)
    val sgi1 = sparseGradInput.toTable[Tensor[Float]](1)

    val a = sparseModel.getParametersTable()
    val b = model.getParametersTable()

    val sparse_grad = sparseModel.getParameters()._2
    val dense_grad = model.getParameters()._2

    val SPS = sparse_grad.storage().array()
    val DEN = dense_grad.storage().array()
    var cnt = 0
    for (i <- SPS.indices) {
      if (SPS(i) != DEN(i)) {
        cnt += 1
        if (cnt % 300 == 0) {
          println(s"SPS(${i})(${SPS(i)}) != DEN(${i})(${DEN(i)})")
        }
      }
    }
    println(s"Total Non-EQUAL: ${cnt}")

    sparseModel.getParameters()._1.equals(model.getParameters()._1) shouldEqual true

    sparseModel.getParameters()._2.equals(model.getParameters()._2) shouldEqual true

    output shouldEqual sparseOutput

    denseGradInput shouldEqual sparseGradInput

    a shouldEqual b
  }

}
