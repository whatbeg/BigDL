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


COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
AGE, WORKCLASS, FNLWGT, EDUCATION, EDUCATION_NUM, MARITAL_STATUS, OCCPATION, \
RELATIONSHIP, RACE, GENDER, CAPITAL_GAIN, CAPITAL_LOSS, HOURS_PER_WEEK, NATIVE_COUNTRY, \
AGE_BUCKETS, LABEL = range(16)

LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def get_data(train_file_name='train.data', test_file_name='test.data'):
    df_train = pd.read_csv(train_file_name,
                           names=COLUMNS,
                           skipinitialspace=True,
                           engine="python")

    df_test = pd.read_csv(test_file_name,
                          names=COLUMNS,
                          skipinitialspace=True,
                          skiprows=1,   # skip first line: "|1x3 Cross Validator"
                          engine="python")

    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    df_train[LABEL_COLUMN] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    # print(df_train)
    # print(df_test)
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


def bucketized_column(column, boundaries):
    """
    transform every value of a column to corresponding bucket according to boundaries

    :param column: primitive column
    :param boundaries: boundaries to bucketize
    :return: bucketized column
    """
    _column = copy.deepcopy(column)
    for i in range(len(_column)):
        _column[i] = binary_search(_column[i], boundaries)
    return _column


def discretize_for_lookupTable(df, data_type, lookup_dict, columns):
    """
    discretize for BigDL's lookupTable's requirement: elements of input should be little than or equal to $nIndex + 1

    :param df: data tensor. Type must be numpy.ndarray
    :param columns: columns to do discretize
    :return: discretized data tensor
    """
    if data_type == 'train':
        for col in columns:
            total = sorted({}.fromkeys(df[:, col]).keys())
            total_dict = {k: i+1
                          for i, k in enumerate(total)}
            for _ in range(len(df[:, col])):
                if df[_, col] not in total_dict.keys():
                    df[_, col] = 1
                else:
                    df[_, col] = total_dict[df[_, col]]
            lookup_dict[col] = total_dict
    elif data_type == 'test':
        for col in columns:
            total_dict = lookup_dict[col]
            for _ in range(len(df[:, col])):
                if df[_, col] not in total_dict.keys():
                    df[_, col] = 1
                else:
                    df[_, col] = total_dict[df[_, col]]
    else:
        raise ValueError("Not valid data type")
    return df, lookup_dict


def feature_columns(df, data_type, lookup_dict):
    gender_dict = {"Male": 1, "Female": 2}
    age_boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    assert WORKCLASS == 1 and EDUCATION == 3 and CAPITAL_LOSS == 11 and NATIVE_COUNTRY == 13
    for i in range(df.shape[0]):
        df[i, WORKCLASS] = (hash(df[i, 1]) % 100 + 100) % 100       # workclass
        df[i, EDUCATION] = (hash(df[i, 3]) % 1000 + 1000) % 1000    # education
        df[i, RELATIONSHIP] = (hash(df[i, 7]) % 100 + 100) % 100       # relationship
        df[i, OCCPATION] = (hash(df[i, 6]) % 1000 + 1000) % 1000    # occupation
        df[i, NATIVE_COUNTRY] = (hash(df[i, 13]) % 1000 + 1000) % 1000  # native_country
        df[i, GENDER] = gender_dict[df[i, 9]] \
            if (df[i, 9] in gender_dict.keys()) else -1  # gender
        df[i, AGE] = df[i, 0]    # age
        df[i, EDUCATION_NUM] = df[i, 4]    # education_num
        df[i, CAPITAL_GAIN] = df[i, 10]    # capital_gain
        df[i, CAPITAL_LOSS] = df[i, 11]    # capital_loss
        df[i, HOURS_PER_WEEK] = df[i, 12]  # hours_per_week
    age_bucket = bucketized_column(df[:, AGE], boundaries=age_boundaries)
    # print(age_bucket)
    df[:, AGE_BUCKETS] = age_bucket
    df, lookup_dict = discretize_for_lookupTable(df, data_type, lookup_dict, columns=[WORKCLASS, EDUCATION, RELATIONSHIP, OCCPATION, NATIVE_COUNTRY, GENDER])
    return df, lookup_dict


def make_wide_deep_columns(df):
    wide_columns = np.array(df[:, GENDER])
    wide_columns = np.c_[wide_columns, df[:, NATIVE_COUNTRY]]
    wide_columns = np.c_[wide_columns, df[:, EDUCATION], df[:, OCCPATION]]
    wide_columns = np.c_[wide_columns, df[:, WORKCLASS], df[:, RELATIONSHIP]]
    wide_columns = np.c_[wide_columns, df[:, AGE_BUCKETS]]

    deep_columns = np.array(df[:, WORKCLASS])
    deep_columns = np.c_[deep_columns, df[:, EDUCATION], df[:, GENDER]]
    deep_columns = np.c_[deep_columns, df[:, RELATIONSHIP], df[:, NATIVE_COUNTRY]]
    deep_columns = np.c_[deep_columns, df[:, OCCPATION]]

    deep_columns = np.c_[deep_columns, df[:, AGE], df[:, EDUCATION_NUM], df[:, CAPITAL_GAIN]]
    deep_columns = np.c_[deep_columns, df[:, CAPITAL_LOSS], df[:, HOURS_PER_WEEK]]

    wide_deep_columns = np.c_[wide_columns, deep_columns]
    print(wide_deep_columns)
    print(wide_deep_columns.shape)
    return wide_deep_columns, np.array(df[:, LABEL])


def handle():
    df_train, df_test = get_data()
    df_train = np.array(df_train)
    df_test = np.array(df_test)
    df_train, lookup_dict = feature_columns(df_train, 'train', {})
    df_test, _ = feature_columns(df_test, 'test', lookup_dict)
    # print(df_train)
    train_data, train_label = make_wide_deep_columns(df_train)
    test_data, test_label = make_wide_deep_columns(df_test)
    np.savetxt("train_tensor.data", train_data, fmt="%d", delimiter=',')
    np.savetxt("train_label.data", train_label, fmt="%d")
    np.savetxt("test_tensor.data", test_data, fmt="%d", delimiter=',')
    np.savetxt("test_label.data", test_label, fmt="%d")

handle()
