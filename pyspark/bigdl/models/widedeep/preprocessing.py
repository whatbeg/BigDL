#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Still in experimental stage!


from optparse import OptionParser
import os
import sys
import copy
import numpy as np
import pandas as pd
import scipy as sp


CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]

AGE, WORKCLASS, FNLWGT, EDUCATION, EDUCATION_NUM, MARITAL_STATUS, OCCPATION, \
RELATIONSHIP, RACE, GENDER, CAPITAL_GAIN, CAPITAL_LOSS, HOURS_PER_WEEK, NATIVE_COUNTRY, \
AGE_BUCKETS, LABEL, EDUCATION_OCCUPATION, AGEBUCKET_EDUCATION_OCCUPATION, NATIVECOUNTRY_OCCUPATION = range(19)

LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def get_data(train_file_name='train.data', test_file_name='test.data'):
    df_train = pd.read_csv(train_file_name,
                           names=CSV_COLUMNS,
                           skipinitialspace=True,
                           engine="python")

    df_test = pd.read_csv(test_file_name,
                          names=CSV_COLUMNS,
                          skipinitialspace=True,
                          skiprows=1,       # skip first line: "|1x3 Cross Validator"
                          engine="python")

    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    df_train[LABEL_COLUMN] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    return df_train, df_test


def binary_search(val, array, start=0):
    """
    binary search implementation

    :param val: value to search
    :param array: data array to be searched
    :param start: 0 if array starts with 0 else 1
    :return: location of val in array, or bucket fall in if not in array
    """
    low = start
    high = len(array) - 1 + start
    while low <= high:
        mid = (low + high) / 2
        if array[mid] == val:
            return mid
        elif array[mid] > val:
            high = mid-1
        else:
            low = mid+1
    return low


def bucketized_column(column, boundaries, start=0):
    """
    transform every value of a column to corresponding bucket according to boundaries

    :param column: primitive column
    :param boundaries: boundaries to bucketize
    :param start: start with 0 or 1
    :return: bucketized column
    """
    _column = copy.deepcopy(column)
    for i in range(len(_column)):
        _column[i] = binary_search(_column[i], boundaries) + start
    return _column


def cross_column(columns, hash_backet_size=1000, start=1):
    """
    generate cross column feature from `columns` with hash bucket.

    :param columns: columns to use to generate cross column, Type must be ndarray
    :param hash_backet_size: hash bucket size to bucketize cross columns to fixed hash bucket
    :return: cross column, represented as a ndarray
    """
    assert columns.shape[0] > 0 and columns.shape[1] > 0
    _crossed_column = np.zeros(columns.shape[0])
    for i in range(columns.shape[0]):
        _crossed_column[i] = (hash("_".join(map(str, columns[i, :]))) % hash_backet_size
                                 + hash_backet_size) % hash_backet_size + start
    return _crossed_column

def categorical_column_with_vocabulary_list(column, vocab_list, default=1):

    n = column.shape[0]
    assert n > 0 and len(vocab_list) > 0
    vocab_dict = {}
    for i, word in enumerate(vocab_list):
        vocab_dict[word] = i+1
    _newcol = np.zeros(n)
    for row in range(n):
        _newcol[row] = vocab_dict[column[row]] if column[row] in vocab_dict else default
    return _newcol

def sparse_column(column, vocab_size):
    """
    convert integer id to sparse representation.
    For example, 3 -> [0, 0, 0, 1, 0, ...]

    :param column: the whole column with integer ids of this feature, Type: ndarray
    :param vocab_size: length of sparse vector
    :return: new column consist of converted sparse features, Type: ndarray
    """
    n = column.shape[0]
    assert n > 0 and vocab_size > 0
    _newcol = np.zeros((n, vocab_size))
    for row in range(n):
        ind = int(column[row])
        # print("ind = {}".format(ind))
        assert 0 < ind <= vocab_size
        np.put(_newcol[row], ind-1, 1)
    return _newcol


def feature_columns(df):

    age_boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    age_bucket = bucketized_column(df[:, AGE], boundaries=age_boundaries, start=1)
    df[:, AGE_BUCKETS] = age_bucket
    df[:, GENDER] = categorical_column_with_vocabulary_list(df[:, GENDER], ["Female", "Male"])
    df[:, EDUCATION] = categorical_column_with_vocabulary_list(df[:, EDUCATION], [   # 16
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
    df[:, MARITAL_STATUS] = categorical_column_with_vocabulary_list(df[:, MARITAL_STATUS], [  # 7
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
    df[:, RELATIONSHIP] = categorical_column_with_vocabulary_list(df[:, RELATIONSHIP], [  #6
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
    df[:, WORKCLASS] = categorical_column_with_vocabulary_list(df[:, WORKCLASS], [    # 9
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

    for i in range(df.shape[0]):
        df[i, OCCPATION] = (hash(df[i, 6]) % 1000 + 1000) % 1000 + 1        # occupation
        df[i, NATIVE_COUNTRY] = (hash(df[i, 13]) % 1000 + 1000) % 1000 + 1  # native_country

    education_occupation = cross_column(df[:, [EDUCATION, OCCPATION]], hash_backet_size=int(1000))
    agebucket_education_occpation = cross_column(df[:, [AGE_BUCKETS, EDUCATION, OCCPATION]], hash_backet_size=int(1000))
    nativecountry_occupation = cross_column(df[:, [NATIVE_COUNTRY, OCCPATION]], hash_backet_size=int(1000))
    df = np.c_[df, education_occupation, nativecountry_occupation, agebucket_education_occpation]
    return df


def make_wide_deep_columns(df):

    base_columns = np.array(df[:, GENDER])
    base_columns = np.c_[base_columns, df[:, EDUCATION], df[:, MARITAL_STATUS], df[:, RELATIONSHIP]]
    base_columns = np.c_[base_columns, df[:, WORKCLASS], sparse_column(df[:, OCCPATION], 1000),
                         sparse_column(df[:, NATIVE_COUNTRY], 1000), df[:, AGE_BUCKETS]]

    crossed_columns = np.array(sparse_column(df[:, EDUCATION_OCCUPATION], 1000))
    crossed_columns = np.c_[crossed_columns, sparse_column(df[:, AGEBUCKET_EDUCATION_OCCUPATION], 1000)]
    crossed_columns = np.c_[crossed_columns, sparse_column(df[:, NATIVECOUNTRY_OCCUPATION], 1000)]

    deep_columns = np.array(sparse_column(df[:, WORKCLASS], 9))
    deep_columns = np.c_[deep_columns, sparse_column(df[:, EDUCATION], 16), sparse_column(df[:, GENDER], 2)]
    deep_columns = np.c_[deep_columns, sparse_column(df[:, RELATIONSHIP], 6)]

    deep_columns = np.c_[deep_columns, df[:, NATIVE_COUNTRY]]   # for embedding 8 dims
    deep_columns = np.c_[deep_columns, df[:, OCCPATION]]        # for embedding 8 dims

    deep_columns = np.c_[deep_columns, df[:, AGE], df[:, EDUCATION_NUM], df[:, CAPITAL_GAIN]]
    deep_columns = np.c_[deep_columns, df[:, CAPITAL_LOSS], df[:, HOURS_PER_WEEK]]

    wide_deep_columns = np.c_[base_columns, crossed_columns, deep_columns]
    return np.c_[wide_deep_columns, np.array(df[:, LABEL])]


def handle():
    df_train, df_test = get_data()
    df_train = np.array(df_train)
    df_test = np.array(df_test)
    df_train = feature_columns(df_train)
    df_test = feature_columns(df_test)

    train_data = make_wide_deep_columns(df_train[:])
    np.savetxt("./data/train_tensor.data", train_data, fmt="%d", delimiter=',')
    del train_data

    test_data = make_wide_deep_columns(df_test[:])
    np.savetxt("./data/test_tensor.data", test_data, fmt="%d", delimiter=',')
    del test_data

handle()
