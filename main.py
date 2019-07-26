# import read_dat
import preprocess
import model
from model import *
import sys
import sim_impl
from sim_impl import *
from multiprocessing import Pool
import ast
from preprocess import *
import os
import rawdata
import pickle
import keras
import ctypes
from ctypes import *


import types
import tempfile
import keras.models

try:
    _est = ctypes.CDLL('./librelationships.so')
    print('Imported .so Library...')
    FER_INVALID = -1.0
    MAX_RATES = 96
except:
    pass


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def get_data(scenario=1):
    return pd.read_csv('data/csv/scenario-0%d-new.csv' % scenario, dtype='float')

def C_test_lstm(RATES, knownfers, points):
    print("Entering the Testing Phase...")
    agency = rawdata.Agency.load(RATES)
    for i in range(1, RATES+1):
        print('testing for ', i)
        # knownfers[i] = 0.20

        max_rates_array_type = ctypes.c_double * (MAX_RATES+1)
        rates_used = max_rates_array_type()
        individual_ests = max_rates_array_type()
        est_fer = c_double(FER_INVALID)
        source_fer = c_double(FER_INVALID)
        actual_fer = c_double(FER_INVALID)
        confidence = c_int(0)
        model.C_test_model(i, knownfers, agency, RATES, points, max_rates_array_type, rates_used, individual_ests, est_fer, source_fer, actual_fer, confidence)
    return agency

def test_lstm(testing_data, fer_cols):
    print("Entering the Testing Phase...")
    agency = rawdata.Agency.load()
    for i in range(1, 97):
        make_keras_picklable()
        print('testing for ', i)
        mod_str = 'models/model%d.h5' % i
        # mod = load_model(mod_str)
        mod = pickle.load(open(mod_str, 'rb'))
        # mod = mod_str
        print('model loaded for ', i)
        # print('Fasttracking this model')
        model.test_model(testing_data, mod, i, fer_cols, agency)
    return agency


def train_all_models(data_train, fer_cols):
    print('Training all models...')
    for x in range(1,97):
        make_keras_picklable()
        print('Training model for ', x)
        rate_features = []
        rate_features.append('R%d_FER' % x)
        # rate_features.append('R%d_RSSI' % x)

        mod = model.train_model(data_train, rate_features, fer_cols, epochs=200)
        # mod.save('models/model%d.h5' % x)
        fname = 'modelsBIG/model%d.h5' % x
        pickle.dump(mod, open(fname, 'wb'))

# def C_SetCoverScore_Parallel(set_cover):
#     for a in agency.agents.keys():
#         agent = agency.agents[a]
#         agent.updateTrustTowards(all_rates)
#         agent.current_time_pred = {}
#         agent.sim_total_good_preds = agent.initPreds()
#         agent.sim_total_pred_from = agent.initPreds()
#     set_cover = sim_impl.C_CalculateSetCoverScore(set_cover, agency, all_predictors, amt_pred)
#     return set_cover


def C_start_sim(agency, points, RATES, knownfers):
    rates_plus_one = RATES + 1
    best_dict = {}
    best = list(range(1,rates_plus_one))
    all_rates = []
    for i in range(1,(RATES+1)):
        all_rates.append(i)

    #loops through for each size
    for current_iter in range(0,len(best)):
        print("In the simultaion...")
        score_dict = {}
        new_sets = preprocess.get_set_covers(best)

        #remembers old best for THRESHOLD
        old_best = best
        max_set_trust_score = 0
        good_preds = 0
        total_preds = 0
        i = 1


        set_cover_objs = rawdata.Covers.load(new_sets)
        if current_iter == 0:
            # print(agency)
            all_predictors, amt_pred = sim_impl.C_run_sim_fast(RATES, agency, points, knownfers)

        pass_arr = []
        for s in set_cover_objs.covers.keys():
            set_cover = set_cover_objs.covers[s]
            pass_arr.append(set_cover)

        # agents = 24
        # chunksize = 4
        #
        # with Pool(processes=agents) as pool:
        #     set_covers_arr = pool.map(C_SetCoverScore_Parallel, pass_arr, chunksize)

        for s in set_cover_objs.covers.keys():
            set_cover = set_cover_objs.covers[s]
            for a in agency.agents.keys():
                agent = agency.agents[a]
                agent.updateTrustTowards(all_rates)
                agent.current_time_pred = {}
                agent.sim_total_good_preds = agent.initPreds()
                agent.sim_total_pred_from = agent.initPreds()
            set_cover = sim_impl.C_CalculateSetCoverScore(set_cover, agency, all_predictors, amt_pred)

        for s in set_cover_objs.covers.keys():
            set_cover = set_cover_objs.covers[s]
        # for set_cover in set_cover_arr:
            if set_cover.score > max_set_trust_score:
                max_set_trust_score = set_cover.score
                best = set_cover.rates
                good_preds = set_cover.good_predictions
                total_preds = set_cover.total_predictions

        print('Best Cover: ', best)
        print('Score: ', max_set_trust_score)
        print('Total Good: ', good_preds)
        print('Total Preds: ', total_preds)
        print("Percent Correct: ", good_preds/total_preds)
        best_dict[len(best)] = [max_set_trust_score, best, good_preds, total_preds]

        if len(best) == 1:
            return best_dict


def start_sim(dataset, agency):
    best_dict = {}
    best = list(range(1,97))
    all_rates = best

    #loops through for each size
    for current_iter in range(0,len(best)):
        print("In the simultaion...")
        score_dict = {}
        new_sets = preprocess.get_set_covers(best)

        #remembers old best for THRESHOLD
        old_best = best
        max_set_trust_score = 0
        good_preds = 0
        total_preds = 0

        i = 1

        set_cover_objs = rawdata.Covers.load(new_sets)
        if current_iter == 0:
            print("Going into run simultaion fast...")
            all_predictors, amt_pred = sim_impl.run_sim_fast(dataset, agency)
            print("Finished all predictions...")

        for s in set_cover_objs.covers.keys():
            set_cover = set_cover_objs.covers[s]
            for a in agency.agents.keys():
                agent = agency.agents[a]
                agent.updateTrustTowards(all_rates)
                agent.current_time_pred = {}
                agent.sim_total_good_preds = agent.initPreds()
                agent.sim_total_pred_from = agent.initPreds()
            # set_cover = sim_impl.run_sim(set_cover, dataset, agency)
            set_cover = sim_impl.CalculateSetCoverScore(set_cover, dataset, agency, all_predictors, amt_pred)

        for s in set_cover_objs.covers.keys():
            set_cover = set_cover_objs.covers[s]
            if set_cover.score > max_set_trust_score:
                max_set_trust_score = set_cover.score
                best = set_cover.rates
                good_preds = set_cover.good_predictions
                total_preds = set_cover.total_predictions

        print('Best Cover: ', best)
        print('Score: ', max_set_trust_score)
        print('Total Good: ', good_preds)
        print('Total Preds: ', total_preds)
        print("Percent Correct: ", good_preds/total_preds)
        best_dict[len(best)] = [max_set_trust_score, best, good_preds, total_preds]

        if len(best) == 1:
            return best_dict


def start():
    if len(sys.argv) == 1:
        print("========Training a new model========")
        # scenario_name = sys.argv[1]
        dataset = [get_data(i) for i in range(1,8)]
        fer_cols = ['R%d_FER' % i for i in range(1, 97)]

        rssi_min, rssi_max = preprocess.get_rssi_min_max(dataset)

        for data in dataset:
            data[rssi_cols] = encode_rssi(data[rssi_cols], rssi_min, rssi_max).fillna(1)

        data_train, data_test, data24_train, data5_train, data24_test, data5_test, data24_sim, data5_sim = preprocess.split_data(dataset)

        train_all_models(data24_train, fer_cols)
        print('Trained all models...')
        exit()

        time_dataset = []
        time_dataset.append(data24_test[0])
        agency = test_lstm(time_dataset, fer_cols)

        sim_dataset = []
        sim_dataset.append(data24_sim[0])
        full_best_dict = start_sim(sim_dataset, agency)

        util_file.best_dict_to_csv(full_best_dict)
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'run':
            print("========Using Existing Models========")
            # scenario_name = sys.argv[1]
            dataset = [get_data(i) for i in range(1,8)]
            fer_cols = ['R%d_FER' % i for i in range(1, 97)]

            rssi_min, rssi_max = preprocess.get_rssi_min_max(dataset)

            for data in dataset:
                data[rssi_cols] = encode_rssi(data[rssi_cols], rssi_min, rssi_max).fillna(1)

            data_train, data_test, data24_train, data5_train, data24_test, data5_test, data24_sim, data5_sim = preprocess.split_data(dataset)

            time_dataset = []
            time_dataset.append(data24_test[2])
            agency = test_lstm(time_dataset, fer_cols)

            sim_dataset = []
            sim_dataset.append(data24_sim[2])
            full_best_dict = start_sim(sim_dataset, agency)

            util_file.best_dict_to_csv(full_best_dict)
        if sys.argv[1] == 'c':
            print("========Using The C Code For Prediction========")
            points = _est.python_init(b"data/1-sec-average-bugFix/twoHourStableS8/s8.cfg",
                 b"s8-1-rates.sets",
                 b"data/1-sec-average-bugFix/twoHourStableS8-a",
                 b"data/1-sec-average-bugFix/twoHourStableS8-b")

            RATES = _est.get_rates()

            rates_array_type = ctypes.c_double * (RATES+1)
            knownfers = rates_array_type()
            for x in range(1, RATES+1):
                knownfers[x] = FER_INVALID  # -1.0 means not used

            agency = C_test_lstm(RATES, knownfers, points)

            full_best_dict = C_start_sim(agency, points, RATES, knownfers)

            util_file.best_dict_to_csv(full_best_dict)

if __name__ == '__main__':
    start()
