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
    var res = Tensor[Float](2)
    SparseTensorMath.addmv[Float](res, 1, Tensor[Float](2).fill(0), 1, sparseM, a)
    val correctRes = Tensor[Float](2)
    correctRes.setValue(1, 7)
    correctRes.setValue(2, 25)

    res shouldEqual correctRes
  }
}
