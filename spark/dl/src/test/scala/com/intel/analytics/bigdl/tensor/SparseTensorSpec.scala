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

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat

class SparseTensorSpec  extends FlatSpec with Matchers {
  "concat" should "return right result on dim-1 concatenation" in {
    val sTensor1 = Tensor.sparse(Tensor(3).range(1, 3, 1))
    val sTensor2 = Tensor.sparse(Tensor(3).range(4, 6, 1))
    val result = Tensor.sparse(Array(2, 3), 6)
    result.concat(1, Array(sTensor1, sTensor2), result)
    val expectedResult = Tensor(2, 3)
    expectedResult.narrow(1, 1, 1).range(1, 3, 1)
    expectedResult.narrow(1, 2, 1).range(4, 6, 1)
    Tensor.dense(result) should be (expectedResult)
  }

  "concat" should "return right result on dim-1 concatenation 2" in {
    val sTensor1 = Tensor.sparse(Tensor(3).setValue(2, 1))
    val sTensor2 = Tensor.sparse(Tensor(3).setValue(3, 3))
    val sTensor3 = Tensor.sparse(Tensor(3).setValue(1, 2))
    val result = Tensor.sparse(Array(3, 3), 3)
    result.concat(1, Array(sTensor1, sTensor2, sTensor3), result)
    val expectedResult = Tensor(3, 3)
    expectedResult.setValue(1, 2, 1)
    expectedResult.setValue(2, 3, 3)
    expectedResult.setValue(3, 1, 2)
    Tensor.dense(result) should be (expectedResult)
  }

  "concat" should "return right result" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 3).range(1, 9, 1))
    val sTensor2 = Tensor.sparse(Tensor(3, 2).range(10, 15, 1))
    val result = Tensor.sparse(Array(3, 5), 15)
    result.concat(2, Array(sTensor1, sTensor2), result)
    val exceptedResult = Tensor(3, 5)
    exceptedResult.narrow(2, 1, 3).range(1, 9, 1)
    exceptedResult.narrow(2, 4, 2).range(10, 15, 1)
    Tensor.dense(result) should be (exceptedResult)
  }

  "concat" should "return right result2" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 3).range(1, 9, 1))
    val sTensor2 = Tensor.sparse(Tensor(3, 2).range(10, 15, 1))
    val sTensor3 = Tensor.sparse(Tensor(3, 2).range(16, 21, 1))
    val result = Tensor.sparse(Array(3, 7), 21)
    result.concat(2, Array(sTensor1, sTensor2, sTensor3), result)
    val exceptedResult = Tensor(3, 7)
    exceptedResult.narrow(2, 1, 3).range(1, 9, 1)
    exceptedResult.narrow(2, 4, 2).range(10, 15, 1)
    exceptedResult.narrow(2, 6, 2).range(16, 21, 1)
    Tensor.dense(result) should be (exceptedResult)
  }

  "concat" should "return right result 3" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 3).setValue(1, 1, 1))
    val sTensor2 = Tensor.sparse(Tensor(3, 2).setValue(1, 2, 2))
    val sTensor3 = Tensor.sparse(Tensor(3, 2).setValue(2, 1, 3))
    val result = Tensor.sparse(Array(3, 7), 3)
    result.concat(2, Array(sTensor1, sTensor2, sTensor3), result)
    val exceptedResult = Tensor(3, 7)
    exceptedResult.setValue(1, 1, 1)
    exceptedResult.setValue(1, 5, 2)
    exceptedResult.setValue(2, 6, 3)
    Tensor.dense(result) should be (exceptedResult)
  }

  "concat" should "return right result on first dimension" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 3).setValue(1, 1, 1))
    val sTensor2 = Tensor.sparse(Tensor(2, 3).setValue(1, 2, 2))
    val sTensor3 = Tensor.sparse(Tensor(2, 3).setValue(2, 1, 3))
    val result = Tensor.sparse(Array(7, 3), 3)
    result.concat(1, Array(sTensor1, sTensor2, sTensor3), result)
    val exceptedResult = Tensor(7, 3)
    exceptedResult.setValue(1, 1, 1)
    exceptedResult.setValue(4, 2, 2)
    exceptedResult.setValue(7, 1, 3)
    Tensor.dense(result) should be (exceptedResult)
  }

  "concat" should "return right result on first dimension 2" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 3).setValue(2, 1, 1)).narrow(1, 2, 2)
    val sTensor2 = Tensor.sparse(Tensor(2, 3).setValue(1, 2, 2))
    val sTensor3 = Tensor.sparse(Tensor(2, 3).setValue(2, 1, 3))
    val result = Tensor.sparse(Array(6, 3), 3)
    result.concat(1, Array(sTensor1, sTensor2, sTensor3), result)
    val exceptedResult = Tensor(6, 3)
    exceptedResult.setValue(1, 1, 1)
    exceptedResult.setValue(3, 2, 2)
    exceptedResult.setValue(6, 1, 3)
    Tensor.dense(result) should be (exceptedResult)
  }
}
