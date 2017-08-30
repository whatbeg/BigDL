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

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.{Sample, TensorSample}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{File, T}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scopt.OptionParser

object Utils {

  case class TrainParams(
    folder: String = "./",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    batchSize: Int = 480,
    learningRate: Double = 0.001,
    learningRateDecay: Double = 0.0,
    maxEpoch: Int = 20,
    coreNumber: Int = -1,
    nodeNumber: Int = -1,
    overWriteCheckpoint: Boolean = false
  )

  val trainParser = new OptionParser[TrainParams]("BigDL Wide and Deep Learning Example") {
    opt[String]('f', "folder")
      .text("where you put the Census data")
      .action((x, c) => c.copy(folder = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]('d', "learningRateDecay")
      .text("learning rate decay")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Unit]("overWrite")
      .text("overwrite checkpoint files")
      .action( (_, c) => c.copy(overWriteCheckpoint = true) )
  }

  case class TestParams(
    folder: String = "./",
    model: String = "",
    batchSize: Int = 480
  )

  val testParser = new OptionParser[TestParams]("BigDL Lenet Test Example") {
    opt[String]('f', "folder")
      .text("where you put the Census data")
      .action((x, c) => c.copy(folder = x))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

  val AGE = 0
  val WORKCLASS = 1
  val FNLWGT = 2
  val EDUCATION = 3
  val EDUCATION_NUM = 4
  val MARITAL_STATUS = 5
  val OCCUPATION = 6
  val RELATIONSHIP = 7
  val RACE = 8
  val GENDER = 9
  val CAPITAL_GAIN = 10
  val CAPITAL_LOSS = 11
  val HOURS_PER_WEEK = 12
  val NATIVE_COUNTRY = 13
  val LABEL = 14

  val LABEL_COLUMN = "label"
  val COLUMNS = Array("age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket")
  val CATEGORICAL_COLUMNS = Array("workclass", "education", "marital_status", "occupation",
  "relationship", "race", "gender", "native_country")
  val CONTINUOUS_COLUMNS = Array("age", "education_num", "capital_gain", "capital_loss",
  "hours_per_week")

  private[this] def getAgeboundaries(age: String, start: Int = 0): Int = {
    if (age == "?") 0 + start
    else {
      val numage = age.toInt
      if (numage < 18) 0 else if (numage < 25) 1 else if (numage < 30) 2 else if (numage < 35) 3
      else if (numage < 40) 4 else if (numage < 45) 5 else if (numage < 50) 6
      else if (numage < 55) 7 else if (numage < 60) 8 else if (numage < 65) 9 else 10
    }
  }

  private[this] def getGender(gen: String = "Male", start: Int = 0): Int = {
    if (gen == "Female") start + 1 else start
  }

  private[this] def hashbucket(sth: String, bucketsize: Int = 1000, start: Int = 0): Int = {
    (sth.hashCode() % bucketsize + bucketsize) % bucketsize + start
  }

  /**
   * Load Train data of Census dataset.
   *
   * @param sc spark context
   * @param featureFile the file name of train data
   * @return
   */
  private[bigdl] def loadTrain(sc: SparkContext, featureFile: String): RDD[Sample[Float]] = {

    var src: RDD[String] = null
    if (featureFile.startsWith(File.hdfsPrefix)) {
      src = sc.textFile(featureFile)
    } else {
      src = sc.textFile(Paths.get(featureFile).toString)
    }
    val iter = src.filter(s => (s.length > 0)).map(_.stripMargin.split(","))
    // println(iter)
    val storage = Storage[Float](16)
    val storageArray = storage.array()
    var i = 0
    val results = iter.map(line => {
      val indices = new Array[Int](16)
      val lis = line.toSeq
      indices(0) = getGender(lis(GENDER), start = 0)                  // 2
      indices(1) = hashbucket(lis(NATIVE_COUNTRY), 1000) + 2          // 1002
      indices(2) = hashbucket(lis(EDUCATION), 1000) + 1002            // 2002
      indices(3) = hashbucket(lis(OCCUPATION), 1000) + 2002            // 3002
      indices(4) = hashbucket(lis(WORKCLASS), 100) + 3002             // 3102
      indices(5) = hashbucket(lis(RELATIONSHIP), 100) + 3102          // 3202
      indices(6) = getAgeboundaries(lis(AGE), start = 0) + 3202       // 3213
      indices(7) = hashbucket(lis(EDUCATION) + lis(OCCUPATION), 10000) + 3213 // 13213
      indices(8) = hashbucket(lis(NATIVE_COUNTRY) + lis(OCCUPATION), 10000) + 13213 // 23213
      indices(9) = hashbucket(
        getAgeboundaries(lis(AGE)).toString + lis(EDUCATION) + lis(OCCUPATION) + 1000000) + 23213
      // 1023213
      for (k <- 0 until 10) storageArray(k) = 1

      indices(10) = 1023213  // workclass
      indices(11) = 1023214  // education
      indices(12) = 1023215  // gender
      indices(13) = 1023216  // relationship
      indices(14) = 1023217  // native_country
      indices(15) = 1023218  // occupation

      storageArray(10) = indices(4)
      storageArray(11) = indices(2)
      storageArray(12) = indices(0)
      storageArray(13) = indices(5)
      storageArray(14) = indices(1)
      storageArray(15) = indices(3)

      val sps = Tensor.sparse(Array(indices), storage, Array(1023219), 1)
      val den = Tensor[Float](T(lis(AGE).toFloat, lis(EDUCATION_NUM).toFloat,
        lis(CAPITAL_GAIN).toFloat, lis(CAPITAL_LOSS).toFloat,
        lis(HOURS_PER_WEEK).toFloat))
      val train_label = if (lis(LABEL) == ">50K") Tensor[Float](T(1.0f))
                        else Tensor[Float](T(0.0f))

      TensorSample[Float](Array(sps, den), Array(train_label))
    })
    results
  }

  /**
   * Load Test data of Census dataset.
   *
   * @param sc spark context
   * @param featureFile the file name of test data
   * @return
   */
  private[bigdl] def loadTest(sc: SparkContext, featureFile: String): RDD[Sample[Float]] = {

    var src: RDD[String] = null
    if (featureFile.startsWith(File.hdfsPrefix)) {
      src = sc.textFile(featureFile)
    } else {
      src = sc.textFile(Paths.get(featureFile).toString)
    }
    val iter = src.filter(s => (s != "|1x3 Cross validator" && s.length > 0))
      .map(_.stripMargin.split(","))

    val storage = Storage[Float](16)
    val storageArray = storage.array()
    val results = iter.map(line => {
      val indices = new Array[Int](16)
      val lis = line.toSeq
      indices(0) = getGender(lis(GENDER), start = 0)                  // 2
      indices(1) = hashbucket(lis(NATIVE_COUNTRY), 1000) + 2          // 1002
      indices(2) = hashbucket(lis(EDUCATION), 1000) + 1002            // 2002
      indices(3) = hashbucket(lis(OCCUPATION), 1000) + 2002           // 3002
      indices(4) = hashbucket(lis(WORKCLASS), 100) + 3002             // 3102
      indices(5) = hashbucket(lis(RELATIONSHIP), 100) + 3102          // 3202
      indices(6) = getAgeboundaries(lis(AGE), start = 0) + 3202       // 3213
      indices(7) = hashbucket(lis(EDUCATION) + lis(OCCUPATION), 10000) + 3213 // 13213
      indices(8) = hashbucket(lis(NATIVE_COUNTRY) + lis(OCCUPATION), 10000) + 13213 // 23213
      indices(9) = hashbucket(
        getAgeboundaries(lis(AGE)).toString + lis(EDUCATION) + lis(OCCUPATION) + 1000000) + 23213
      // 1023213
      for (k <- 0 until 10) storageArray(k) = 1

      indices(10) = 1023213  // workclass
      indices(11) = 1023214  // education
      indices(12) = 1023215  // gender
      indices(13) = 1023216  // relationship
      indices(14) = 1023217  // native_country
      indices(15) = 1023218  // occupation

      storageArray(10) = indices(4)
      storageArray(11) = indices(2)
      storageArray(12) = indices(0)
      storageArray(13) = indices(5)
      storageArray(14) = indices(1)
      storageArray(15) = indices(3)

      val sps = Tensor.sparse(Array(indices), storage, Array(1023219), 1)
      val den = Tensor[Float](T(lis(AGE).toFloat, lis(EDUCATION_NUM).toFloat,
        lis(CAPITAL_GAIN).toFloat, lis(CAPITAL_LOSS).toFloat,
        lis(HOURS_PER_WEEK).toFloat))
      val train_label = if (lis(LABEL) == ">50K") Tensor[Float](T(1.0f))
      else Tensor[Float](T(0.0f))

      TensorSample[Float](Array(sps, den), Array(train_label))
    })
    results
  }
}
