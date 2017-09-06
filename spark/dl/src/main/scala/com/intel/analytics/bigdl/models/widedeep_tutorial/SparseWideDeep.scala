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

package com.intel.analytics.bigdl.models.widedeep_tutorial

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


object SparseWideDeep {
  def apply[T: ClassTag](modelType: String, classNum: Int = 2)
                        (implicit ev: TensorNumeric[T]): Module[T] = {
    val model = Sequential()
    val wideModel = Sequential().add(Narrow(2, 9, 3000)).add(Reshape(Array(3000)))
    val onlywideModel = Identity()
    val deepModel = Sequential()
    val deepColumn = Concat(2)
    // indicator columns
    deepColumn.add(Sequential().add(Narrow(2, 1, 33)).add(Reshape(Array(33))
      .setName("indicator")))
    // native_country 1000
    deepColumn.add(Sequential().add(Select(2, 34)).add(LookupTable(1000, 8, 0.0)
      .setName("embedding_1")))
    // occupation 1000
    deepColumn.add(Sequential().add(Select(2, 35)).add(LookupTable(1000, 8, 0.0)
      .setName("embedding_2")))
    // numeric column
    deepColumn.add(Sequential().add(Narrow(2, 36, 5)).add(Reshape(Array(5))))
    deepModel.add(deepColumn).add(Linear(54, 100).setName("fc_1"))
      .add(ReLU()).add(Linear(100, 50).setName("fc_2")).add(ReLU())
    modelType match {
      case "wide_n_deep" =>
        val parallel = ParallelTable()
        parallel.add(wideModel)
        parallel.add(deepModel.add(ToSparse()))
        model.add(parallel).add(SparseJoinTable(2))
          .add(SparseLinear(3050, classNum, backwardStart = 3001, backwardLength = 50)
            .setName("fc_3")).add(LogSoftMax())
      case "wide" =>
        model.add(onlywideModel)
          .add(SparseLinear(3008, classNum).setName("fc_3")).add(LogSoftMax())
      case "deep" =>
        deepModel.add(Linear(50, classNum).setName("fc_3")).add(LogSoftMax())
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }
}
