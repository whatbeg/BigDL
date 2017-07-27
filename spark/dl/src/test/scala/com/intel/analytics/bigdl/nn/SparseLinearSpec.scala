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

package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

class SparseLinearSpec extends FlatSpec with Matchers {
  "Sparse Linear" should "return the same result with Linear" in {
    val weight = Tensor.range(1, 8, 1).resize(2, 4)
    val bias = Tensor(2)
    val gradOutput = Tensor.range(1, 4, 1).resize(2, 2)
    val sl = SparseLinear(4, 2)
    val l = Linear(4, 2)
    l.weight.copy(weight)
    l.bias.copy(bias)
    sl.weight.copy(weight)
    sl.bias.copy(bias)
    val input = Tensor(2, 4)
    input.setValue(1, 1, 1f)
    input.setValue(2, 3, 3f)
    val sparseInput = Tensor.toSparse(input)
    val out1 = sl.forward(sparseInput)
    sl.backward(input, gradOutput)
    val out2 = l.forward(input)
    l.backward(input, gradOutput)
    out1 should be (out2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }

  "Sparse Linear" should "return the same result with Linear 2" in {
    val gradOutput = Tensor(2, 2).rand()
    val input = Tensor(2, 4).rand()
    val sl = SparseLinear(4, 2)
    val l = Linear(4, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val sparseInput = Tensor.toSparse(input)
    val out1 = sl.forward(sparseInput)
    sl.backward(input, gradOutput)
    val out2 = l.forward(input)
    l.backward(input, gradOutput)
    out1 should be (out2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }


  "Sparse Linear" should "return the same result with Linear 3" in {
    val gradOutput = Tensor(2, 2).rand()
    val input = Tensor(2, 4).rand()
    val sl = SparseLinear(4, 2, backwardStart = 0, backwardLength = 4)
    val l = Linear(4, 2)
    l.weight.copy(sl.weight)
    l.bias.copy(sl.bias)
    val sparseInput = Tensor.toSparse(input)
    val out1 = sl.forward(sparseInput)
    val gradInput1 = sl.backward(sparseInput, gradOutput)
    val out2 = l.forward(input)
    val gradInput2 = l.backward(input, gradOutput)
    out1 should be (out2)
    gradInput1 should be (gradInput2)
    sl.getParameters()._2 should be (l.getParameters()._2)
  }
}
