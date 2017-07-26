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

  def coogemv[T: ClassTag](
        alpha: T,
        mat: Tensor[T],
        vec: Tensor[T],
        beta: T,
        r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    require(mat.isInstanceOf[SparseTensor[T]] && mat.isContiguous())
    require(vec.isInstanceOf[DenseTensor[T]] && vec.isContiguous())
    require(r.isInstanceOf[DenseTensor[T]] && r.isContiguous())

    (alpha, mat, vec, beta, r)  match {
      case (alpha: Double, a: SparseTensor[Double], x: DenseTensor[Double],
      beta: Double, y: DenseTensor[Double]) =>
        coodgemv(alpha, a, x, beta, y)
      case (alpha: Float, a: SparseTensor[Float], x: DenseTensor[Float],
      beta: Float, y: DenseTensor[Float]) =>
        coosgemv(alpha, a, x, beta, y)
      case _ =>
        throw new IllegalArgumentException(s"Sparse addmv doesn't support")
    }
  }

  private def coodgemv(
        alpha: Double,
        A: SparseTensor[Double],
        x: DenseTensor[Double],
        beta: Double,
        y: DenseTensor[Double]): Unit = {
    val xValues = x.storage().array()
    val yValues = y.storage().array()
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val Arows = A._indices(A.indices_order(0))
    val Acols = A._indices(A.indices_order(1))

    if (beta != 1) {
      MKL.vdscal(yValues.size, beta, yValues, 0, 1)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < Avals.length) {
      val Arow = Arows(valueCounter)
      val Acol = Acols(valueCounter)
      val Aval = Avals(valueCounter)
      yValues(Arow-1) += Aval * alpha * xValues(Acol-1)
      valueCounter += 1
    }
  }

  private def coosgemv(
                alpha: Float,
                A: SparseTensor[Float],
                x: DenseTensor[Float],
                beta: Float,
                y: DenseTensor[Float]): Unit = {
    val xValues = x.storage().array()
    val yValues = y.storage().array()
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val Arows = A._indices(A.indices_order(0))
    val Acols = A._indices(A.indices_order(1))

    if (beta != 1) {
      MKL.vsscal(yValues.size, beta, yValues, 0, 1)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < Avals.length) {
      val Arow = Arows(valueCounter)
      val Acol = Acols(valueCounter)
      val Aval = Avals(valueCounter)
      yValues(Arow-1) += Aval * alpha * xValues(Acol-1)
      valueCounter += 1
    }
  }

  def coogemm[T: ClassTag](
                            alpha: T,
                            mat1: Tensor[T],
                            mat2: Tensor[T],
                            beta: T,
                            r: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    require(mat1.isInstanceOf[SparseTensor[T]] && mat1.isContiguous())
    require(mat2.isInstanceOf[DenseTensor[T]] && mat2.isContiguous())
    require(r.isInstanceOf[DenseTensor[T]] && r.isContiguous())

    var _r: Tensor[T] = null
    var _m1: Tensor[T] = mat1
    var _m2: Tensor[T] = mat2
    var transpose_r = ' '
    if (r.stride(1) == 1 && r.stride(2) != 0) {
      transpose_r = 'n'
      _r = r
    } else if (r.stride(2) == 1 && r.stride(1) != 0) {
      val swap = _m2
      _m2 = _m1
      _m1 = swap
      transpose_r = 't'
      _r = r
    } else {
      transpose_r = 'n'
      _r = new DenseTensor[T](r.size(2), r.size(1))
      _r.copy(r)
      _r = _r.transpose(1, 2)
    }

    val index1 = if (transpose_r == 'n') 1 else 2
    val index2 = if (transpose_r == 'n') 2 else 1
    var transpose_m1 = ' '
    var __m1: Tensor[T] = null
    if (_m1.stride(index1) == 1 && _m1.stride(index2) != 0) {
      transpose_m1 = 'n'
      __m1 = _m1
    } else if (_m1.stride(index2) == 1 && _m1.stride(index1) != 0) {
      transpose_m1 = 't'
      __m1 = _m1
    } else {
      transpose_m1 = if (transpose_r == 'n') 't' else 'n'
      __m1 = _m1.contiguous()
    }

    var transpose_m2 = ' '
    var __m2: Tensor[T] = null
    if (_m2.stride(index1) == 1 && _m2.stride(index2) != 0) {
      transpose_m2 = 'n'
      __m2 = _m2
    } else if (_m2.stride(index2) == 1 && _m2.stride(index1) != 0) {
      transpose_m2 = 't'
      __m2 = _m2
    } else {
      transpose_m2 = if (transpose_r == 'n') 't' else 'n'
      __m2 = _m2.contiguous()
    }

    (alpha, mat1, mat2, beta, r)  match {
//      case (alpha: Double, a: SparseTensor[Double], x: DenseTensor[Double],
//      beta: Double, y: DenseTensor[Double]) =>
//        coodgemm(alpha, a, x, beta, y)
      case (alpha: Float, a: SparseTensor[Float], x: DenseTensor[Float],
      beta: Float, y: DenseTensor[Float]) =>
        coosgemm(alpha, a, x, beta, y)
      case _ =>
        throw new IllegalArgumentException(s"Sparse addmv doesn't support")
    }
  }

  private def coosgemm(
        alpha: Float,
        A: SparseTensor[Float],
        B: DenseTensor[Float],
        beta: Float,
        C: DenseTensor[Float]): Unit = {
//    val xValues = x.storage().array()
//    val yValues = y.storage().array()
//    val mA: Int = A._shape(0)
//    val nA: Int = A._shape(1)
//    val k: Int = x.size(2)
//
//    val Avals = A._values.array()
//    val Arows = A._indices(A.indices_order(0))
//    val Acols = A._indices(A.indices_order(1))

    val mA: Int = A._shape(0)
    val nB: Int = B.size(2)
    val kA: Int = A._shape(1)
    val kB: Int = B.size(1)

    val Avals = A._values.array()
    val Bvals = B.storage().array()
    val Cvals = C.storage().array()
    val ArowIndices = A._indices(A.indices_order(0))
    val AcolPtrs = A._indices(A.indices_order(1))

    // Slicing is easy in this case. This is the optimal multiplication setting for sparse matrices
    if (A.isTransposed){
      var colCounterForB = 0
      if (!B.isTransposed) { // Expensive to put the check inside the loop
        while (colCounterForB < nB) {
          var rowCounterForA = 0
          val Cstart = colCounterForB * mA
          val Bstart = colCounterForB * kA
          while (rowCounterForA < mA) {
            var i = AcolPtrs(rowCounterForA)
            val indEnd = AcolPtrs(rowCounterForA + 1)
            var sum = 0.0
            while (i < indEnd) {
              sum += Avals(i) * Bvals(Bstart + ArowIndices(i))
              i += 1
            }
            val Cindex = Cstart + rowCounterForA
            Cvals(Cindex) = beta * Cvals(Cindex) + sum * alpha
            rowCounterForA += 1
          }
          colCounterForB += 1
        }
      } else {
        while (colCounterForB < nB) {
          var rowCounterForA = 0
          val Cstart = colCounterForB * mA
          while (rowCounterForA < mA) {
            var i = AcolPtrs(rowCounterForA)
            val indEnd = AcolPtrs(rowCounterForA + 1)
            var sum = 0.0
            while (i < indEnd) {
              sum += Avals(i) * B(ArowIndices(i), colCounterForB)
              i += 1
            }
            val Cindex = Cstart + rowCounterForA
            Cvals(Cindex) = beta * Cvals(Cindex) + sum * alpha
            rowCounterForA += 1
          }
          colCounterForB += 1
        }
      }
    } else {
      // Scale matrix first if `beta` is not equal to 0.0
      if (beta != 0.0) {
        f2jBLAS.dscal(C.values.length, beta, C.values, 1)
      }
      // Perform matrix multiplication and add to C. The rows of A are multiplied by the columns of
      // B, and added to C.
      var colCounterForB = 0 // the column to be updated in C
      if (!B.isTransposed) { // Expensive to put the check inside the loop
        while (colCounterForB < nB) {
          var colCounterForA = 0 // The column of A to multiply with the row of B
          val Bstart = colCounterForB * kB
          val Cstart = colCounterForB * mA
          while (colCounterForA < kA) {
            var i = AcolPtrs(colCounterForA)
            val indEnd = AcolPtrs(colCounterForA + 1)
            val Bval = Bvals(Bstart + colCounterForA) * alpha
            while (i < indEnd) {
              Cvals(Cstart + ArowIndices(i)) += Avals(i) * Bval
              i += 1
            }
            colCounterForA += 1
          }
          colCounterForB += 1
        }
      } else {
        while (colCounterForB < nB) {
          var colCounterForA = 0 // The column of A to multiply with the row of B
          val Cstart = colCounterForB * mA
          while (colCounterForA < kA) {
            var i = AcolPtrs(colCounterForA)
            val indEnd = AcolPtrs(colCounterForA + 1)
            val Bval = B(colCounterForA, colCounterForB) * alpha
            while (i < indEnd) {
              Cvals(Cstart + ArowIndices(i)) += Avals(i) * Bval
              i += 1
            }
            colCounterForA += 1
          }
          colCounterForB += 1
        }
      }
    }
  }

}
