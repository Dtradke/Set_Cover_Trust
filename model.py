import math
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.pipeline import make_pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from keras.layers import Dense, LSTM, Dropout, Activation, Flatten, TimeDistributed, Reshape
from bisect import bisect
from collections import Counter
from tqdm import tqdm_notebook
from multiprocessing import Process
import json
from multiprocessing import Pool
import rawdata
from main import *
import util_file
import random
import ctypes
from ctypes import *

THRESHOLD = .1
try:
    _est = ctypes.CDLL('./librelationships.so')
    FER_INVALID = -1.0
    MAX_RATES = 96
except:
    pass

class FERSequence(Sequence):
    def __init__(self, dataset, time_step, batch_size, skip_step, features, targets, shuffle=None):
        self.dataset, self.time_step, self.batch_size = dataset, time_step, batch_size
        self.skip_step, self.features, self.targets = skip_step, features, targets
        self.time_frames = [(len(data) - self.time_step + 1) for data in self.dataset]
        self.data_count = [(time_frames + self.skip_step - 1) // self.skip_step for time_frames in self.time_frames]
        self.data_cum = np.cumsum(self.data_count)
        if shuffle:
            new_dataset = []
            for data in dataset:
                data_c = data.copy()
                col = data_c[shuffle].values
                np.random.shuffle(col)
                data_c[shuffle] = col
                new_dataset.append(data_c)
            self.dataset = new_dataset

    def __len__(self):
        return int(np.ceil(self.data_cum[-1] / self.batch_size))

    def get_single_item(self, idx):
        dataset_idx = bisect(self.data_cum, idx)
        # print(dataset_idx)
        # exit()
        in_dataset_idx = idx
        if dataset_idx > 0:
            in_dataset_idx -= self.data_cum[dataset_idx - 1]
        x = self.dataset[dataset_idx].loc[in_dataset_idx * self.skip_step:in_dataset_idx * self.skip_step + self.time_step - 1, self.features]
        y = self.dataset[dataset_idx].loc[in_dataset_idx * self.skip_step:in_dataset_idx * self.skip_step + self.time_step - 1, self.targets]
        return x.values, y.values

    def __getitem__(self, idx):
        indices = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        indices = [i % self.data_cum[-1] for i in indices]
        batch_x = np.ndarray(shape=(self.batch_size, self.time_step, len(self.features)))
        batch_y = np.ndarray(shape=(self.batch_size, self.time_step, len(self.targets)))
        for i, ind in enumerate(indices):
            x, y = self.get_single_item(ind)
            batch_x[i, :, :] = x
            batch_y[i, :, :] = y
        return batch_x, batch_y



def train_model(dataset, features, targets, hidden_size=96, time_step=10, skip_step=1, batch_size=30, epochs=50, dropout=0.25, save=False):
    train_generator = FERSequence(dataset, time_step, batch_size, skip_step, features, targets)
    model = Sequential()
    model.add(LSTM(units=len(targets), unroll=True, return_sequences=True, recurrent_dropout=dropout, input_shape=(time_step, len(features))))
    #model.add(Dropout(dropout))
    #model.add(LSTM(units=hidden_size, return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(TimeDistributed(Dense(len(targets), activation='relu')))
    #model.add(LSTM(units=len(targets), unroll=True, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, input_shape=(time_step, len(features))))
    #model.add(Dense(len(targets), activation='relu'))
    model.add(Activation('relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    callbacks = [ModelCheckpoint(filepath='checkpoint/model-{epoch:04d}.hdf5', verbose=1, period=1)] if save else []
    #model.summary()
    model.fit_generator(train_generator, epochs=epochs, callbacks=callbacks, use_multiprocessing=True, workers=80, shuffle=True)
    return model


def test_model(dataset, mod, predictor, fer_cols, agency, time_step=10):
    feature_str = 'R%d_FER' % predictor
    features = [feature_str]

    for target in range(1,97):
        agent = agency.agents[target]




        target_str = 'R%d_FER' % target
        target_rssi = 'R%d_RSSI' % target
        target_arr = [target_str]

        train_generator = FERSequence(dataset, time_step, 1, 1, features, target_arr)

        if predictor == agent.id:
            agent.trust_towards[predictor] = 1
            agent.pre_sim_trust[predictor] = 1
            continue
#fastrack
        # else:
        #     for i in range(0,train_generator.time_frames[0]):
        #         agent.total_pred_from[predictor] += 1
        #         prediction = i / 100
        #         target_ground = dataset[0][target_str][i + 9]
        #         error = util_file.findError(prediction, target_ground)
        #         if error < THRESHOLD:
        #             agent.total_good_preds[predictor] += 1
        #         # agent.trust_towards[predictor] = .1#random.uniform(0, 1)
        #         agent.pre_sim_trust[predictor] = .1
        #     agent.trust_towards[predictor] = agent.total_good_preds[predictor] / agent.total_pred_from[predictor]
        #     agent.pre_sim_trust[predictor] = agent.total_good_preds[predictor] / agent.total_pred_from[predictor]
        #     # agent.total_pred_from[predictor] += 10
        #     # agent.total_good_preds[predictor] += 1
        #     continue
#fastrack
        idx = 0

        for i in range(0,train_generator.time_frames[0]):
            x, t = train_generator.get_single_item(i)
            y = mod.predict(np.array([x]))[0]
            agent.total_pred_from[predictor] += 1
            cur_time = y[-1:]
            prediction = cur_time[0][target - 1]
            target_ground = dataset[0][target_str][idx + 9]
            error = util_file.findError(prediction, target_ground)

            if error < THRESHOLD:
                agent.total_good_preds[predictor] += 1

            idx += 1

        agent.trust_towards[predictor] = agent.total_good_preds[predictor] / agent.total_pred_from[predictor]
        agent.pre_sim_trust[predictor] = agent.total_good_preds[predictor] / agent.total_pred_from[predictor]


def C_test_model(predictor, knownfers, agency, RATES, points, max_rates_array_type, rates_used, individual_ests, est_fer, source_fer, actual_fer, confidence):

    for target in range(1,RATES+1):
        agent = agency.agents[target]
        if predictor == agent.id:
            agent.trust_towards[predictor] = 1
            agent.pre_sim_trust[predictor] = 1
            continue

        source_rate = predictor
        toestimate = target
        timesteps = points // 3

        for y in range(1, RATES):
            knownfers[y] = FER_INVALID

        # Very short/simple test
        for point in range(0, timesteps):
            agent.total_pred_from[predictor] += 1
            _est.python_get_rate_fer(source_rate, point, byref(source_fer));
            knownfers[source_rate] = source_fer

            # Now do the estimate
            _est.python_estimate(knownfers, toestimate, points, byref(est_fer), rates_used, individual_ests, byref(confidence))
            _est.python_get_rate_fer(toestimate, point, byref(actual_fer));
            # print("Estimate for rate", toestimate, "by rate ", source_rate, "with fer",
            #       source_fer.value, "at point", point, "=", est_fer.value,
            #       "actual fer = ", actual_fer.value)


            prediction = est_fer.value
            target_ground = actual_fer.value
            error = util_file.findError(prediction, target_ground)
            if error < THRESHOLD:
                agent.total_good_preds[predictor] += 1

        agent.trust_towards[predictor] = agent.total_good_preds[predictor] / agent.total_pred_from[predictor]
        agent.pre_sim_trust[predictor] = agent.total_good_preds[predictor] / agent.total_pred_from[predictor]
