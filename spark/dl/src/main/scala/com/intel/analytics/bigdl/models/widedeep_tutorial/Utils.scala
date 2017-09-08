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
  val CSV_COLUMNS = Array("age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket")
  val CATEGORICAL_COLUMNS = Array("workclass", "education", "marital_status", "occupation",
  "relationship", "race", "gender", "native_country")
  val CONTINUOUS_COLUMNS = Array("age", "education_num", "capital_gain", "capital_loss",
  "hours_per_week")
  val education_vocab = Array("Bachelors", "HS-grad", "11th", "Masters", "9th",
  "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
  "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
  "Preschool", "12th") // 16
  val marital_status_vocab = Array("Married-civ-spouse", "Divorced", "Married-spouse-absent",
    "Never-married", "Separated", "Married-AF-spouse", "Widowed")
  val relationship_vocab = Array("Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
    "Other-relative")  // 6
  val workclass_vocab = Array("Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
    "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked") // 9
  val gender_vocab = Array("Female", "Male")

  private[this] def getAgeboundaries(age: String, start: Int = 0): Int = {
    if (age == "?") 0 + start
    else {
      val numage = age.toInt
      if (numage < 18) 0 else if (numage < 25) 1 else if (numage < 30) 2 else if (numage < 35) 3
      else if (numage < 40) 4 else if (numage < 45) 5 else if (numage < 50) 6
      else if (numage < 55) 7 else if (numage < 60) 8 else if (numage < 65) 9 else 10
    }
  }

  private[this] def hashbucket(sth: String, bucketsize: Int = 1000, start: Int = 0): Int = {
    (sth.hashCode() % bucketsize + bucketsize) % bucketsize + start
  }

  private[this] def categoricalFromVocabList(sth: String,
    vocab_list: Array[String], default: Int = 1, start: Int = 0): Int = {
    start + (if (vocab_list.contains(sth)) vocab_list.indexOf(sth) else default)
  }

  /**
   * Load data of Census dataset.
   *
   * @param sc spark context
   * @param featureFile the file name of train data
   * @param tag "Train" or "Test", represents train data or test data
   * @return
   */
  private[bigdl] def load(sc: SparkContext,
    featureFile: String, tag: String = "Train"): RDD[Sample[Float]] = {

    var src: RDD[String] = null
    if (featureFile.startsWith(File.hdfsPrefix)) {
      src = sc.textFile(featureFile)
    } else {
      src = sc.textFile(Paths.get(featureFile).toString)
    }
    val iter = if (tag == "Train") src.filter(s => (s.length > 0)).map(_.trim().split(","))
    else src.filter(s => (!s.contains("|1x3 Cross validator") && s.length > 0))
      .map(_.trim().split(","))

    val storage = Storage[Float](11)
    val storageArray = storage.array()
    val results = iter.map(line => {
      val indices = new Array[Int](11)
      val lis = line.map(_.trim()).toSeq
      for (k <- 0 until 5) indices(k) = k
      indices(5) = hashbucket(lis(OCCUPATION), 1000, start = 0) + 5
      indices(6) = hashbucket(lis(NATIVE_COUNTRY), 1000, start = 0) + 1005
      indices(7) = 2005
      indices(8) = hashbucket(lis(EDUCATION) + lis(OCCUPATION), 1000, start = 0) + 2006 // 2006
      indices(9) = hashbucket(
        getAgeboundaries(lis(AGE)).toString + lis(EDUCATION) + lis(OCCUPATION), 1000) + 3006 // 2006
      indices(10) = hashbucket(lis(NATIVE_COUNTRY) + lis(OCCUPATION), 1000) + 4006 // 4006

      // 5006
      storageArray(0) = categoricalFromVocabList(lis(GENDER), gender_vocab, default = -1, start = 0)
      storageArray(1) = categoricalFromVocabList(
        lis(EDUCATION), education_vocab, default = -1, start = 0)
      storageArray(2) = categoricalFromVocabList(
        lis(MARITAL_STATUS), marital_status_vocab, default = -1, start = 0)
      storageArray(3) = categoricalFromVocabList(
        lis(RELATIONSHIP), relationship_vocab, default = -1, start = 0)
      storageArray(4) = categoricalFromVocabList(
        lis(WORKCLASS), workclass_vocab, default = -1, start = 0)

      storageArray(5) = 1
      storageArray(6) = 1
      storageArray(7) = getAgeboundaries(lis(AGE), 0)

      for (k <- 8 until 11) storageArray(k) = 1

      val sps = Tensor.sparse(Array(indices), storage, Array(5006), 11)
      val den = Tensor[Float](40).fill(0)
      den.setValue(
        categoricalFromVocabList(lis(WORKCLASS), workclass_vocab, start = 1), 1
      ) // 9
      den.setValue(
        categoricalFromVocabList(lis(EDUCATION), education_vocab, start = 10), 1
      ) // 16
      den.setValue(
        categoricalFromVocabList(lis(GENDER), gender_vocab, start = 26), 1
      ) // 2
      den.setValue(
        categoricalFromVocabList(lis(RELATIONSHIP), relationship_vocab,
          start = 28), 1
      ) // 6
      // total : 33
      den.setValue(34, hashbucket(lis(NATIVE_COUNTRY), 1000, 1).toFloat)
      den.setValue(35, hashbucket(lis(OCCUPATION), 1000, 1).toFloat)
      den.setValue(36, lis(AGE).toFloat)
      den.setValue(37, lis(EDUCATION_NUM).toFloat)
      den.setValue(38, lis(CAPITAL_GAIN).toFloat)
      den.setValue(39, lis(CAPITAL_LOSS).toFloat)
      den.setValue(40, lis(HOURS_PER_WEEK).toFloat)
      den.resize(1, 40)
      val train_label = if (lis(LABEL).contains(">50K")) Tensor[Float](T(2.0f))
                        else Tensor[Float](T(1.0f))
      train_label.resize(1, 1)

      TensorSample[Float](Array(sps, den), Array(train_label))
    })
    results
  }

  private[bigdl] def load2(sc: SparkContext,
                          featureFile: String, tag: String = "Train"): RDD[Array[Tensor[Float]]] = {

    var src: RDD[String] = null
    if (featureFile.startsWith(File.hdfsPrefix)) {
      src = sc.textFile(featureFile)
    } else {
      src = sc.textFile(Paths.get(featureFile).toString)
    }
    val iter = if (tag == "Train") src.filter(s => (s.length > 0)).map(_.trim().split(","))
    else src.filter(s => (!s.contains("|1x3 Cross validator") && s.length > 0))
      .map(_.trim().split(","))

    val storage = Storage[Float](11)
    val storageArray = storage.array()
    val results = iter.map(line => {
      val indices = new Array[Int](11)
      val lis = line.map(_.trim()).toSeq
      for (k <- 0 until 5) indices(k) = k
      indices(5) = hashbucket(lis(OCCUPATION), 1000, start = 0) + 5
      indices(6) = hashbucket(lis(NATIVE_COUNTRY), 1000, start = 0) + 1005
      indices(7) = 2005
      indices(8) = hashbucket(lis(EDUCATION) + lis(OCCUPATION), 1000, start = 0) + 2006 // 2006
      indices(9) = hashbucket(
        getAgeboundaries(lis(AGE)).toString + lis(EDUCATION) + lis(OCCUPATION), 1000) + 3006 // 2006
      indices(10) = hashbucket(lis(NATIVE_COUNTRY) + lis(OCCUPATION), 1000) + 4006 // 4006

      // 5006
      storageArray(0) = categoricalFromVocabList(lis(GENDER), gender_vocab, default = -1, start = 0)
      storageArray(1) = categoricalFromVocabList(
        lis(EDUCATION), education_vocab, default = -1, start = 0)
      storageArray(2) = categoricalFromVocabList(
        lis(MARITAL_STATUS), marital_status_vocab, default = -1, start = 0)
      storageArray(3) = categoricalFromVocabList(
        lis(RELATIONSHIP), relationship_vocab, default = -1, start = 0)
      storageArray(4) = categoricalFromVocabList(
        lis(WORKCLASS), workclass_vocab, default = -1, start = 0)

      storageArray(5) = 1
      storageArray(6) = 1
      storageArray(7) = getAgeboundaries(lis(AGE), 0)

      for (k <- 8 until 11) storageArray(k) = 1

      val sps = Tensor.sparse(Array(indices), storage, Array(5006), 11)
      val den = Tensor[Float](40).fill(0)
      den.setValue(
        categoricalFromVocabList(lis(WORKCLASS), workclass_vocab, start = 1), 1
      ) // 9
      den.setValue(
        categoricalFromVocabList(lis(EDUCATION), education_vocab, start = 10), 1
      ) // 16
      den.setValue(
        categoricalFromVocabList(lis(GENDER), gender_vocab, start = 26), 1
      ) // 2
      den.setValue(
        categoricalFromVocabList(lis(RELATIONSHIP), relationship_vocab,
          start = 28), 1
      ) // 6
      // total : 33
      den.setValue(34, hashbucket(lis(NATIVE_COUNTRY), 1000, 1).toFloat)
      den.setValue(35, hashbucket(lis(OCCUPATION), 1000, 1).toFloat)
      den.setValue(36, lis(AGE).toFloat)
      den.setValue(37, lis(EDUCATION_NUM).toFloat)
      den.setValue(38, lis(CAPITAL_GAIN).toFloat)
      den.setValue(39, lis(CAPITAL_LOSS).toFloat)
      den.setValue(40, lis(HOURS_PER_WEEK).toFloat)
      den.resize(1, 40)
      val train_label = if (lis(LABEL).contains(">50K")) Tensor[Float](T(2.0f))
      else Tensor[Float](T(1.0f))
      train_label.resize(1, 1)

      Array(sps, den, train_label)
    })
    results
  }

}
