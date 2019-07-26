import pickle
import time
import random

import model
from model import *
import util_file
import keras
import ctypes
from ctypes import *

import types
import tempfile
import keras.models

AGENT_NUM = 96
THRESHOLD = .1
try:
    _est = ctypes.CDLL('./librelationships.so')
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


def make_agent_list(set_cover):

    current_trust_dict = {}
    agent_list = []
    initial_val = 1 / len(set_cover)
    for agent in range(1, 97):
        temp_dict = {}
        for s in set_cover:
            temp_dict[s] = initial_val

        single_agent = Agent(agent, temp_dict)
        agent_list.append(single_agent)
    return agent_list

def C_simTrustCalculations(agency, all_predictors, amt_pred, set_cover):
    for time in range(0, amt_pred):
        for a in agency.agents.keys():
            agent = agency.agents[a]
            if agent.id in set_cover.rates:
                continue
            agent.set_cover = set_cover.rates
            for p in all_predictors.predictors.keys():
                predictor = all_predictors.predictors[p]
                target_ground = predictor.target_ground[agent.id][time]
                if predictor.id in set_cover.rates:
                    agent.current_time_pred[predictor.id] = predictor.predictions[agent.id][time]

            prediction = agent.findWeightedPred()
            error = util_file.findError(prediction, agent.ground_truth[time])
            set_cover.total_predictions += 1

            if error < THRESHOLD:
                agent.updatePredictors(set_cover.rates)
                set_cover.good_predictions += 1
            else:
                agent.updateBadPredictors(set_cover.rates)
    return agency, set_cover

def simTrustCalculations(dataset, agency, all_predictors, amt_pred, set_cover):
    rows = dataset[0].shape[0] - 9
    for time in range(0, rows):
        for a in agency.agents.keys():
            agent = agency.agents[a]
            if agent.id in set_cover.rates:
                continue
            target_str = 'R%d_FER' % agent.id
            agent.set_cover = set_cover.rates
            for p in all_predictors.predictors.keys():
                predictor = all_predictors.predictors[p]
                if predictor.id in set_cover.rates:
                    agent.current_time_pred[predictor.id] = predictor.predictions[agent.id][time]

            prediction = agent.findWeightedPred()
            target_ground = dataset[0][target_str][time + 9]
            error = util_file.findError(prediction, target_ground)
            set_cover.total_predictions += 1
            if error < THRESHOLD:
                agent.updatePredictors(set_cover.rates)
                set_cover.good_predictions += 1
            else:
                agent.updateBadPredictors(set_cover.rates)
    return agency, set_cover

def C_run_sim_fast(rates, agency, points, knownfers):
    flag = False
    all_rates = []
    for i in range(1,(rates+1)):
        all_rates.append(i)

    all_predictors = rawdata.AllPredictors.load(rates, all_rates)
    # amt_pred = 0
    timesteps_start = points // 3
    timesteps = points
    amt_pred = timesteps - timesteps_start

    # for a in agency.agents.keys():
    #     agent = agency.agents[a]
    #     agent.updateTrustTowards(all_rates)
    #     agent.current_time_pred = {}
    #     agent.sim_total_good_preds = agent.initPreds()
    #     agent.sim_total_pred_from = agent.initPreds()


    for p in all_predictors.predictors.keys():
        predictor = all_predictors.predictors[p]
        print('predictor: ', predictor.id, " is predicting")

        # knownfers[predictor.id] = 0.20
        max_rates_array_type = ctypes.c_double * (MAX_RATES+1)
        rates_used = max_rates_array_type()
        individual_ests = max_rates_array_type()
        est_fer = c_double(FER_INVALID)
        source_fer = c_double(FER_INVALID)
        actual_fer = c_double(FER_INVALID)
        confidence = c_int(0)

        for target in range(1,rates+1):
            agent = agency.agents[target]
            if predictor == agent.id:
                agent.trust_towards[predictor] = 1
                agent.pre_sim_trust[predictor] = 1
                continue


            source_rate = predictor.id
            toestimate = target


            for y in range(1, rates+1):
                knownfers[y] = FER_INVALID

            # Very short/simple test
            for point in range(timesteps_start, timesteps):
                agent.total_pred_from[predictor.id] += 1
                _est.python_get_rate_fer(source_rate, point, byref(source_fer));
                knownfers[source_rate] = source_fer

                # Now do the estimate
                _est.python_estimate(knownfers, toestimate, points, byref(est_fer), rates_used, individual_ests, byref(confidence))
                _est.python_get_rate_fer(toestimate, point, byref(actual_fer));
                # print("Estimate for rate", toestimate, "by rate ", source_rate, "with fer",
                #       source_fer.value, "at point", point, "=", est_fer.value,
                #       "actual fer = ", actual_fer.value)

                target_ground = actual_fer.value
                if flag is not True:
                    agent.ground_truth.append(target_ground)
                if predictor.id == agent.id:
                    prediction = target_ground
                else:
                    prediction = est_fer.value
                predictor.predictions[target].append(prediction)
                predictor.sim_predictions[target].append(prediction) #new
                predictor.target_ground[target].append(target_ground)
            # print(agent.ground_truth[:10])
            # print(agent.id, " ", len(agent.ground_truth), " ", agent.ground_truth[0])
            # print(timesteps - timesteps_start)

        flag = True

    return all_predictors, amt_pred


def run_sim_fast(dataset, agency, time_step=10):
    all_rates = []
    for i in range(1,97):
        all_rates.append(i)

    all_predictors = rawdata.AllPredictors.load(97, all_rates)
    # all_predictors = rawdata.AllPredictors.load(all_rates)
    amt_pred = 0


    # for a in agency.agents.keys():
    #     agent = agency.agents[a]
    #     agent.updateTrustTowards(all_rates)
    #     agent.current_time_pred = {}
    #     agent.sim_total_good_preds = agent.initPreds()
    #     agent.sim_total_pred_from = agent.initPreds()


    for p in all_predictors.predictors.keys():
        predictor = all_predictors.predictors[p]

#uncomment for fastrack
        make_keras_picklable()
        mod_str = 'models/model%d.h5' % predictor.id
        mod = pickle.load(open(mod_str, 'rb'))
#uncomment for fastrack

        feature_str = 'R%d_FER' % predictor.id
        features = [feature_str]

        for target in range(1,97):
            agent = agency.agents[target]
            print("Predictor: ", predictor.id, " Target: ", agent.id)

#fastrack
            # for x in range(0, 10):
            #     predictor.predictions[target].append(random.uniform(0, 1))
            #     predictor.sim_predictions[target].append(random.uniform(0, 1)) #new
            # continue
#fastrack

            target_str = 'R%d_FER' % target
            target_rssi = 'R%d_RSSI' % target
            target_arr = [target_str]

            train_generator = model.FERSequence(dataset, time_step, 1, 1, features, target_arr)

#fastrack
            # for i in range(0,train_generator.time_frames[0]):
            #     agent.total_pred_from[predictor.id] += 1
            #     prediction = i / 100
            #     predictor.predictions[target].append(prediction)
            #     predictor.sim_predictions[target].append(prediction)
            # amt_pred = i
            # continue
#fastrack

            idx = 0
            for i in range(0,train_generator.time_frames[0]):
                x, t = train_generator.get_single_item(i)
                y = mod.predict(np.array([x]))[0]
                agent.total_pred_from[predictor.id] += 1
                cur_time = y[-1:]
                prediction = cur_time[0][target - 1]
                predictor.predictions[target].append(prediction)
                predictor.sim_predictions[target].append(prediction) #new
                idx += 1

            amt_pred = idx
    return all_predictors, amt_pred

def C_CalculateSetCoverScore(set_cover, agency, all_predictors, amt_pred):
    agency, set_cover = C_simTrustCalculations(agency, all_predictors, amt_pred, set_cover)
    set_cover_score = agency.findSetCoverScore(set_cover)
    set_cover.score = set_cover_score
    return set_cover

def CalculateSetCoverScore(set_cover, dataset, agency, all_predictors, amt_pred):
    agency, set_cover = simTrustCalculations(dataset, agency, all_predictors, amt_pred, set_cover)
    set_cover_score = agency.findSetCoverScore(set_cover)
    set_cover.score = set_cover_score
    return set_cover

#
# def run_sim(set_cover, dataset, agency, time_step=10):
#
#     all_predictors = rawdata.AllPredictors.load(set_cover.rates)
#     amt_pred = 0
#
#     for a in agency.agents.keys():
#         agent = agency.agents[a]
#         agent.updateTrustTowards(set_cover.rates)
#         agent.current_time_pred = {}
#         agent.sim_total_good_preds = agent.initPreds()
#         agent.sim_total_pred_from = agent.initPreds()
#
#
#     for p in all_predictors.predictors.keys():
#         predictor = all_predictors.predictors[p]
#
#         make_keras_picklable()
#         mod_str = 'models/model%d.h5' % predictor.id
#         mod = pickle.load(open(mod_str, 'rb'))
#         # # mod = load_model(mod_str)
#         #
#         feature_str = 'R%d_FER' % predictor.id
#         features = [feature_str]
#
#         for target in range(1,97):
#             agent = agency.agents[target]
#             if target in set_cover.rates:
#                 agent.trust_towards[agent.id] = 1
#                 agent.total_good_preds[agent.id] = agent.total_pred_from[agent.id]
#                 continue
# #fastrack
#             # for x in range(0, 10):
#             #     predictor.predictions[target].append(.1) #random.uniform(0, 1)
# #fastrack
#
#             if len(predictor.predictions[target]) > amt_pred:
#                 amt_pred = len(predictor.predictions[target])
#             continue
#
#             target_str = 'R%d_FER' % target
#             target_rssi = 'R%d_RSSI' % target
#             target_arr = [target_str]
#
#             train_generator = model.FERSequence(dataset, time_step, 1, 1, features, target_arr)
#
#             idx = 0
#
#             for i in range(0,train_generator.time_frames[0]):
#                 x, t = train_generator.get_single_item(i)
#                 y = mod.predict(np.array([x]))[0]
#                 agent.total_pred_from[predictor] += 1
#                 cur_time = y[-1:]
#                 prediction = cur_time[0][target - 1]
#                 predictor.predictions[target].append(prediction)
#                 idx += 1
#
#             # idx = 0
#             # for d, count in tqdm_notebook(list(zip(dataset, train_generator.time_frames)), desc='Evaluate'):
#             #     lst = []
#             #     for _ in tqdm_notebook(range(count - 10), leave=False, desc='Sub-Evaluate'):
#             #         x, t = train_generator.get_single_item(idx)
#             #         y = mod.predict(np.array([x]))[0]
#             #         agent.total_pred_from[predictor] += 1
#             #         cur_time = y[-1:]
#             #         prediction = cur_time[0][target - 1]
#             #         predictor.predictions[target].append(prediction)
#             #         idx += 1
#             #     total = 0
#
#
#     agency, set_cover = simTrustCalculations(dataset, agency, all_predictors, amt_pred, set_cover)
#     set_cover_score = agency.findSetCoverScore()
#     set_cover.score = set_cover_score
#     return set_cover
