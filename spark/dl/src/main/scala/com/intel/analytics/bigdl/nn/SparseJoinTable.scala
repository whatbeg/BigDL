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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class SparseJoinTable[T: ClassTag] (
    val dimension: Int)(implicit ev: TensorNumeric[T])
    extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    var size: Array[Int] = null

    var i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      if (i == 1) {
        size = currentOutput.size()
      } else {
        size(dimension - 1) += currentOutput.size(dimension)
      }
      i += 1
    }
    output.resize(size)

    var offset = 1
    i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      output.narrow(dimension, offset, currentOutput.size(dimension))
        .copy(currentOutput)
      offset += currentOutput.size(dimension)
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var i = 1
    while (i <= input.length()) {
      gradInput(i) = gradOutput
      i += 1
    }
    gradInput
  }

}
