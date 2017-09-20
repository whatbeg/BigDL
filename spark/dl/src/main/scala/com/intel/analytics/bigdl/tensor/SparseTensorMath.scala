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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object SparseTensorMath {

  def addmv[T: ClassTag](
        r : Tensor[T],
        beta : T,
        t : Tensor[T],
        alpha : T,
        mat : Tensor[T],
        vec : Tensor[T])(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    require(mat.nDimension() == 2 && vec.nDimension() == 1)
    require(mat.size(2) == vec.size(1))
    require(t.nDimension() == 1)
    require(t.size(1) == mat.size(1))
    if(!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    SparseTensorBLAS.coogemv(alpha, mat, vec, beta, r)
    r
  }

  // res = beta * mat3 + alpha * mat1 * mat2
  def addmm[T: ClassTag](
         res: Tensor[T],
         beta: T,
         mat3: Tensor[T],
         alpha: T,
         mat1: Tensor[T],
         mat2: Tensor[T]
         )(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    require(mat1.dim() == 2 && mat2.dim() == 2 && mat3.dim() == 2)
    require(mat3.size(1) == mat1.size(1) && mat3.size(2) == mat2.size(2))
    if(!res.eq(mat3)) {
      res.resizeAs(mat3).copy(mat3)
    }

    SparseTensorBLAS.coogemm(alpha, mat, vec, beta, r)
    r
  }
}
