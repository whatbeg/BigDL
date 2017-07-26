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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{SparseTensorMath, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SparseLinear[T: ClassTag](
      inputSize: Int,
      outputSize: Int,
      val withBias: Boolean = true,
      var wRegularizer: Regularizer[T] = null,
      var bRegularizer: Regularizer[T] = null,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null)(implicit ev: TensorNumeric[T]) extends TensorModule[T]{
  require(outputSize == 1, "outputSize should be 1 only")

  val weight: Tensor[T] = Tensor[T](inputSize)
  val bias: Tensor[T] = Tensor[T](outputSize)
  val addBuffer: Tensor[T] = Tensor[T]()
  val gradWeight = Tensor[T](inputSize)
  val gradBias = Tensor[T](outputSize)
  reset()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    output.resize(Array(input.size(1)))
    output.fill(bias.valueAt(1))
    SparseTensorMath.addmv(output, ev.fromType[Int](1), output, ev.fromType[Int](1), input, weight)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    // TODO: implement updateGradInput
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    if (input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
    }
    gradInput
  }

  override def accGradParameters(
                                  input: Tensor[T],
                                  gradOutput: Tensor[T]): Unit = {
    SparseTensorMath.addmv(gradWeight, ev.fromType(0.0), gradWeight,
      ev.fromType[Double](scaleW), input.t(), gradOutput)
    gradBias.add(ev.times(ev.fromType[Double](scaleB), gradOutput.sum()))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def toString() : String = {
    s"nn.SparseLinear($inputSize -> $outputSize)"
  }
}

object SparseLinear {
  def apply[@specialized(Float, Double) T: ClassTag](
          inputSize: Int,
          outputSize: Int,
          withBias: Boolean = true,
          wRegularizer: Regularizer[T] = null,
          bRegularizer: Regularizer[T] = null,
          initWeight: Tensor[T] = null,
          initBias: Tensor[T] = null,
          initGradWeight: Tensor[T] = null,
          initGradBias: Tensor[T] = null
        )(implicit ev: TensorNumeric[T]): SparseLinear[T] = {
    new SparseLinear[T](inputSize, outputSize,
      withBias, wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
