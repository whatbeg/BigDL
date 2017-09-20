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

import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class SparseTensorMathSpec extends FlatSpec with Matchers {
  "Sparse Matrix * Dense Vector" should "be correct" in {
    val sparseMatrix: Tensor[Float] = new DenseTensor[Float](2, 3)
    sparseMatrix.range(1, 12, 2)
    val sparseM = Tensor.sparse(sparseMatrix)
    val a = Tensor[Float](3)
    a.setValue(1, 2)
    a.setValue(3, 1)
    val res = Tensor[Float](2)
    SparseTensorMath.addmv[Float](res, 1, Tensor[Float](2).fill(0), 1, sparseM, a)
    val correctRes = Tensor[Float](2)
    correctRes.setValue(1, 7)
    correctRes.setValue(2, 25)

    res shouldEqual correctRes
  }

  "Sparse Matrix * Dense Matrix" should "be correct" in {
    val sparseMatrix: Tensor[Float] = new DenseTensor[Float](2, 3)
    sparseMatrix.setValue(1, 3, 1)
    sparseMatrix.setValue(2, 2, 1)
    val sparseM = Tensor.sparse(sparseMatrix)
    val denseM = Tensor[Float](2, 3).range(1, 12, 2).t()

    val res = Tensor[Float](2, 2)
    SparseTensorMath.addmm[Float](res, 1, res, 1, sparseM, denseM)
    val correctRes = Tensor[Float](2, 2)
    correctRes.setValue(1, 1, 5)
    correctRes.setValue(1, 2, 11)
    correctRes.setValue(2, 1, 3)
    correctRes.setValue(2, 2, 9)

    res shouldEqual correctRes
  }

  "Dense Matrix * Sparse Matrix" should "be correct" in {
    val sparseMatrix: Tensor[Float] = new DenseTensor[Float](3, 2)
    sparseMatrix.setValue(2, 2, 1)
    sparseMatrix.setValue(3, 1, 1)
    val sparseM = Tensor.sparse(sparseMatrix)
    val denseM = Tensor[Float](2, 3).range(1, 12, 2)

    val res = Tensor[Float](2, 2)
    SparseTensorMath.addmm[Float](res, 1, res, 1, denseM, sparseM)
    val correctRes = Tensor[Float](2, 2)
    correctRes.setValue(1, 1, 5)
    correctRes.setValue(1, 2, 3)
    correctRes.setValue(2, 1, 11)
    correctRes.setValue(2, 2, 9)

    res shouldEqual correctRes
  }
}
