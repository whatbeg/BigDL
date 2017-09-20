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

import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{SparseTensorBLAS, SparseTensorMath, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SparseLinear[T: ClassTag](
      inputSize: Int,
      outputSize: Int,
      backwardStart: Int = -1,
      backwardLength: Int = -1,
      withBias: Boolean = true,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null)(implicit ev: TensorNumeric[T]) extends Linear[T](
  inputSize, outputSize, withBias, wRegularizer, bRegularizer,
  initWeight, initBias, initGradWeight, initGradBias) {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    val nFrame = input.size(1)
    val nElement = output.nElement
    val t = Array(nFrame, weight.size(1))
    output.resize(t)
    if (output.nElement() != nElement) {
      output.zero()
    }

    if (addBuffer.nElement() != nFrame) {
      addBuffer.resize(Array(nFrame)).fill(ev.one)
    }

    SparseTensorBLAS.coomm(ev.one, input, weight.t, ev.zero, output)
    if (withBias) output.addr(ev.one, addBuffer, bias)
    output
  }

  // just backward a part of the gradOutput.
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)
    if (backwardStart >= 0 && backwardLength > 0) {
      // TODO: _input to dense
      val _inputSize = Array(input.size(1), backwardLength)
      val _weight = weight.narrow(2, backwardStart, backwardLength)

      val nElement = gradInput.nElement()
      gradInput.resize(_inputSize)
      if (nElement != gradInput.nElement()) {
        gradInput.zero()
      }

      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, _weight)
    }
    gradInput
  }

  override def accGradParameters(
                                  input: Tensor[T],
                                  gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    gradWeight.resize(outputSize, inputSize)
    if (withBias) {
      gradBias.resize(outputSize)
    }

    if (scaleW != 0) {
      SparseTensorMath.addmm(gradWeight, ev.one, gradWeight,
        ev.fromType[Double](scaleW), gradOutput.t, input)
    }

    if (withBias && scaleB != 0) {
      gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t, addBuffer)
    }

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
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
          backwardStart: Int = -1,
          backwardLength: Int = -1,
          wRegularizer: Regularizer[T] = null,
          bRegularizer: Regularizer[T] = null,
          initWeight: Tensor[T] = null,
          initBias: Tensor[T] = null,
          initGradWeight: Tensor[T] = null,
          initGradBias: Tensor[T] = null
        )(implicit ev: TensorNumeric[T]): SparseLinear[T] = {
    new SparseLinear[T](inputSize, outputSize, backwardStart, backwardLength,
      withBias, wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
