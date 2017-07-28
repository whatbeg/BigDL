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

package com.intel.analytics.bigdl.models.widedeep

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object WideDeep {
  def apply[T: ClassTag](modelType: String, classNum: Int = 2)
                        (implicit ev: TensorNumeric[T]): Module[T] = {
    val model = Sequential()
    val wideModel = Concat(2)
    wideModel.add(Sequential().add(Narrow(2, 1, 1023213)).add(Reshape(Array(1023213))))
    val deepModel = Sequential()
    val deepColumn = Concat(2)
      // workclass 100
    deepColumn.add(Sequential().add(Select(2, 1023214)).add(LookupTable(100, 8, 0.0)))
      // education 1000
    deepColumn.add(Sequential().add(Select(2, 1023215)).add(LookupTable(1000, 8, 0.0)))
       // gender 2
    deepColumn.add(Sequential().add(Select(2, 1023216)).add(LookupTable(2, 8, 0.0)))
       // relationship 100
    deepColumn.add(Sequential().add(Select(2, 1023217)).add(LookupTable(100, 8, 0.0)))
      // native_country 1000
    deepColumn.add(Sequential().add(Select(2, 1023218)).add(LookupTable(1000, 8, 0.0)))
      // occupation 1000
    deepColumn.add(Sequential().add(Select(2, 1023219)).add(LookupTable(1000, 8, 0.0)))
    deepColumn.add(Sequential().add(Narrow(2, 1023220, 5)).add(Reshape(Array(5))))
    deepModel.add(deepColumn).add(Linear(53, 100)).add(ReLU()).add(Linear(100, 50)).add(ReLU())
    modelType match {
      case "wide_n_deep" =>
        wideModel.add(deepModel)
        model.add(wideModel).add(Linear(1023263, classNum)).add(LogSoftMax())
      case "wide" =>
        model.add(wideModel).add(Linear(1023213, classNum)).add(LogSoftMax())
      case "deep" =>
        model.add(deepModel).add(Linear(50, classNum)).add(LogSoftMax())
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }
}

object WideDeepWithSparse {
  def apply[T: ClassTag](modelType: String, classNum: Int = 2)
                        (implicit ev: TensorNumeric[T]): Module[T] = {
    val model = Sequential()
    val wideModel = Identity()
    val deepModel = Sequential()
    val deepColumn = Concat(2)
    // workclass 100
    deepColumn.add(Sequential().add(Select(2, 1023214)).add(LookupTable(100, 8, 0.0)))
    // education 1000
    deepColumn.add(Sequential().add(Select(2, 1023215)).add(LookupTable(1000, 8, 0.0)))
    // gender 2
    deepColumn.add(Sequential().add(Select(2, 1023216)).add(LookupTable(2, 8, 0.0)))
    // relationship 100
    deepColumn.add(Sequential().add(Select(2, 1023217)).add(LookupTable(100, 8, 0.0)))
    // native_country 1000
    deepColumn.add(Sequential().add(Select(2, 1023218)).add(LookupTable(1000, 8, 0.0)))
    // occupation 1000
    deepColumn.add(Sequential().add(Select(2, 1023219)).add(LookupTable(1000, 8, 0.0)))
    deepColumn.add(Sequential().add(Narrow(2, 1023220, 5)).add(Reshape(Array(5))))
    deepModel.add(deepColumn).add(Linear(53, 100)).add(ReLU()).add(Linear(100, 50)).add(ReLU())
    modelType match {
      case "wide_n_deep" =>
        val parallel = ParallelTable()
        parallel.add(wideModel)
        parallel.add(deepModel.add(new ToSparse()))
        model.add(parallel)
          // .add(SparseJoinTable(2, 2))
          .add(SparseLinear(1023263, classNum)).add(LogSoftMax())
      case "wide" =>
        model.add(wideModel).add(Linear(1023213, classNum)).add(LogSoftMax())
      case "deep" =>
        deepModel.add(Linear(50, classNum)).add(LogSoftMax())
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }
}
