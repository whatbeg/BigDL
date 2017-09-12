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

  output = Tensor.sparse(Array(1, 1), 1)

  override def updateOutput(input: Table): Tensor[T] = {
    var size: Array[Int] = null
    var nElements = 0

    var i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      if (i == 1) {
        size = currentOutput.size()
      } else {
        size(dimension - 1) += currentOutput.size(dimension)
      }
      nElements += currentOutput.nElement()
      // print(s"${currentOutput.nElement()} + ")
      i += 1
    }
    output.resize(size, nElements)
    // println(s"joinTable = " + nElements)
    output.concat(2, input, output)

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

object SparseJoinTable {
  def apply[@specialized(Float, Double) T: ClassTag](
        dimension: Int)(implicit ev: TensorNumeric[T]) : SparseJoinTable[T] = {
    new SparseJoinTable[T](dimension)
  }
}
