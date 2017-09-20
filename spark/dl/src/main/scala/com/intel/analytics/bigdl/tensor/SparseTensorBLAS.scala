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

package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object SparseTensorBLAS {

  def coomv[T: ClassTag](
        alpha: T,
        mat: Tensor[T],
        vec: Tensor[T],
        beta: T,
        r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    if (ev.isEq(beta, ev.zero)) {
      r.zero()
    }

    (alpha, mat, vec, beta, r)  match {
      case (alpha: Double, a: SparseTensor[Double], x: DenseTensor[Double],
      beta: Double, y: DenseTensor[Double]) =>
        coodmv(alpha, a, x, beta, y)
      case (alpha: Float, a: SparseTensor[Float], x: DenseTensor[Float],
      beta: Float, y: DenseTensor[Float]) =>
        coosmv(alpha, a, x, beta, y)
      case _ =>
        throw new IllegalArgumentException(s"Sparse addmv doesn't support")
    }
  }

  private def coodmv(
        alpha: Double,
        A: SparseTensor[Double],
        x: DenseTensor[Double],
        beta: Double,
        y: DenseTensor[Double]): Unit = {
    val xValues = x.storage().array()
    val xOffset = x.storageOffset() - 1
    val yValues = y.storage().array()
    val yOffset = y.storageOffset() - 1
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val Arows = A._indices(A.indices_order(0))
    val Acols = A._indices(A.indices_order(1))

    if (beta != 1.0) {
      y.mul(beta)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < Avals.length) {
      val Arow = Arows(valueCounter)
      val Acol = Acols(valueCounter)
      val Aval = Avals(valueCounter)
      yValues(Arow + yOffset) += Aval * alpha * xValues(Acol + xOffset)
      valueCounter += 1
    }
  }

  private def coosmv(
                alpha: Float,
                A: SparseTensor[Float],
                x: DenseTensor[Float],
                beta: Float,
                y: DenseTensor[Float]): Unit = {
    val xValues = x.storage().array()
    val xOffset = x.storageOffset() - 1
    val yValues = y.storage().array()
    val yOffset = y.storageOffset() - 1
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val Arows = A._indices(A.indices_order(0))
    val Acols = A._indices(A.indices_order(1))

    if (beta != 1.0) {
      y.mul(beta)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < Avals.length) {
      val Arow = Arows(valueCounter)
      val Acol = Acols(valueCounter)
      val Aval = Avals(valueCounter)
      yValues(Arow) += Aval * alpha * xValues(Acol)
      valueCounter += 1
    }
  }

  def coomm[T: ClassTag](
        alpha: T,
        mat1: Tensor[T],
        mat2: Tensor[T],
        beta: T,
        r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    if (ev.isEq(beta, ev.zero)) {
      r.zero()
    }

    (alpha, mat1, mat2, beta, r)  match {
//      case (alpha: Double, a: SparseTensor[Double], x: DenseTensor[Double],
//      beta: Double, y: DenseTensor[Double]) =>
//        coodgemm(alpha, a, x, beta, y)
      case (alpha: Float, a: SparseTensor[Float], x: DenseTensor[Float],
      beta: Float, y: DenseTensor[Float]) =>
        coosmm(alpha, a, x, beta, y)
      case (alpha: Float, a: DenseTensor[Float], x: SparseTensor[Float],
      beta: Float, y: DenseTensor[Float]) =>
        coosmm(alpha, a, x, beta, y)
      case _ =>
        throw new IllegalArgumentException(s"Sparse addmm doesn't support")
    }
  }

  private def coosmm(
        alpha: Float,
        A: SparseTensor[Float],
        B: DenseTensor[Float],
        beta: Float,
        C: DenseTensor[Float]): Unit = {
    val mA: Int = A._shape(0)
    val nB: Int = B.size(2)
    val kA: Int = A._shape(1)
    val kB: Int = B.size(1)

    val Avals = A._values.array()
    val Bvals = B.storage().array()
    val bOffset = B.storageOffset() - 1
    val Cvals = C.storage().array()
    val cOffset = C.storageOffset() - 1
    val ArowIndices = A._indices(A.indices_order(0))
    val AcolIndices = A._indices(A.indices_order(1))

    require(ArowIndices.length == AcolIndices.length, s"A: row indices number " +
      s"${ArowIndices.length()} is not equal to col indices number ${AcolIndices.length()}")
    require(ArowIndices.length == Avals.length, s"A: indices length ${ArowIndices.length()}" +
      s"is not equal to values length ${Avals.length}")

    // Scale matrix first if `beta` is not equal to 0.0
    if (beta != 1.0) {
      C.mul(beta)
    }
    // Perform matrix multiplication and add to C. The rows of A are multiplied by the columns of
    // B, and added to C.
    var index = 0
    if (B.stride(2) == 1 && B.size(2) == B.stride(1)) {
      while (index < Avals.length) {
        val curMA = ArowIndices(index)
        val curKA = AcolIndices(index)
        var n = 0
        while (n < nB) {
          Cvals(curMA * nB + n) += Avals(index) * Bvals(curKA * nB + n + bOffset)
          n += 1
        }
        index += 1
      }
    } else {
      while (index < Avals.length) {
        val curMA = ArowIndices(index)
        val curKA = AcolIndices(index)
        var n = 0
        while (n < nB) {
          Cvals(curMA * nB + n + cOffset) += Avals(index) * Bvals(curKA + n * kB + bOffset)
          n += 1
        }
        index += 1
      }

    }
  }

  private def coosmm(
        alpha: Float,
        A: DenseTensor[Float],
        B: SparseTensor[Float],
        beta: Float,
        C: DenseTensor[Float]): Unit = {
    val kB: Int = B.size(1)
    val nB: Int = B.size(2)
    val mA: Int = A.size(1)
    val kA: Int = A.size(2)

    val Bvals = B._values.array()
    val Avals = A.storage().array()
    val aOffset = A.storageOffset() - 1
    val Cvals = C.storage().array()
    val cOffset = C.storageOffset() - 1
    val BrowIndices = B._indices(B.indices_order(0))
    val BcolIndices = B._indices(B.indices_order(1))

    require(BrowIndices.length == BcolIndices.length, s"B: row indices number " +
      s"${BrowIndices.length()} is not equal to col indices number ${BcolIndices.length()}")
    require(BrowIndices.length == Bvals.length, s"B: indices length ${BrowIndices.length()}" +
      s"is not equal to values length ${Bvals.length}")

    // Scale matrix first if `beta` is not equal to 0.0
    if (beta != 1.0) {
      C.mul(beta)
    }
    // Perform matrix multiplication and add to C. The rows of B are multiplied by the columns of
    // A, and added to C.
    var index = 0
    if (A.stride(2) == 1 && A.size(2) == A.stride(1)) {
      while (index < Bvals.length) {
        val curKB = BrowIndices(index)
        val curNB = BcolIndices(index)
        var n = 0
        while (n < mA) {
          Cvals(n * nB + curNB) += Bvals(index) * Avals(n * kA + curKB + aOffset)
          n += 1
        }
        index += 1
      }
    } else {
      while (index < Bvals.length) {
        val curKB = BrowIndices(index)
        val curNB = BcolIndices(index)
        var n = 0
        while (n < mA) {
          Cvals(n * nB + curNB + cOffset) += Bvals(index) * Avals(n + mA * curKB + aOffset)
          n += 1
        }
        index += 1
      }

    }
  }

}
