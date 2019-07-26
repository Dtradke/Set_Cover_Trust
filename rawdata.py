import statistics



class Covers(object):

    def __init__(self, covers):
        self.covers = covers

    @staticmethod
    def load(set_covers):
        covers = {n:SetCover(set_covers[n]) for n in range(0, len(set_covers))}
        return Covers(covers)

    def __repr__(self):
        return "Covers({})".format(self.covers)

class SetCover(object):
    def __init__(self, rates):
        self.rates = rates
        self.score = 0
        self.good_predictions = 0
        self.total_predictions = 0

    def __repr__(self):
        return "SetCover(Score: {},{}, total good: {}, total predictions: {})".format(self.score, self.rates, self.good_predictions, self.total_predictions)





class AllPredictors(object):

    def __init__(self, predictors):
        self.predictors = predictors

    @staticmethod
    def load(rates, set_cover):
        predictors = {n:Predictor(n, rates) for n in set_cover}
        return AllPredictors(predictors)

    def __repr__(self):
        return "AllPredictors({})".format(self.predictors)

class Predictor(object):
    def __init__(self, id, rates=96, predictions=None, test_predictions=None, sim_predictions=None):
        self.id = id
        self.rates = rates
        self.predictions = predictions if predictions is not None else self.initPredictions()
        self.test_prediction = test_predictions if test_predictions is not None else self.initPredictions()
        self.sim_predictions = sim_predictions if sim_predictions is not None else self.initPredictions()
        self.target_ground = self.initPredictions()

    def initPredictions(self):
        init_pred_dict = {}
        for i in range(1, self.rates+1):
            init_pred_dict[i] = []
        return init_pred_dict

    def __repr__(self):
        return "Predictor(id: {}, {})".format(self.id, len(self.predictions[1]))


class Agency(object):

    def __init__(self, agents):
        self.agents = agents

    def findSetCoverScore(self, set_cover):
        max_trusts = []
        for a in self.agents.keys():
            agent = self.agents[a]
            cur_max = 0
            for t in set_cover.rates:#agent.trust_towards.keys():
                if agent.trust_towards[t] > cur_max: # and t != agent.id
                    cur_max = agent.trust_towards[t]
            max_trusts.append(cur_max)
        return statistics.mean(max_trusts)


    @staticmethod
    def load(rates=96):
        agents = {n:Agent(n, rates) for n in range(1,rates+1)}
        return Agency(agents)

    def __repr__(self):
        return "Agency(Agents: {})".format(self.agents)


class Agent( object ):

    def __init__(self, id = None, rates=None, trust_towards = None, good_preds=None, total_preds=None, sim_good_preds=None, sim_total_preds=None, set_cover = None): #set_cover_trust is a dictionary
        self.id = id
        self.trust_towards = trust_towards if trust_towards is not None else self.loadTrustDict(rates)
        self.pre_sim_trust = trust_towards if trust_towards is not None else self.loadTrustDict(rates)
        self.total_good_preds = good_preds if good_preds is not None else self.initPreds(rates)
        self.total_pred_from = total_preds if total_preds is not None else self.initPreds(rates)
        self.sim_total_good_preds = sim_good_preds if good_preds is not None else self.initPreds(rates)
        self.sim_total_pred_from = sim_total_preds if total_preds is not None else self.initPreds(rates)
        self.set_cover = set_cover
        self.current_time_pred = {}
        self.ground_truth = []


    def initPreds(self, rates=96):
        init_pred_dict = {}
        for i in range(1, rates+1):
            init_pred_dict[i] = 0
        return init_pred_dict

    def updateTrustTowards(self, set_cover):
        trust_dict = {}
        old_trust = self.pre_sim_trust
        self.trust_towards = {}
        for i in set_cover:
            self.trust_towards[i] = old_trust[i]

    def loadTrustDict(self, rates=96):
        init_trust_dict = {}
        for i in range(1, rates+1):
            if i == self.id:
                init_trust_dict[i] = 1
            else:
                init_trust_dict[i] = 0
        return init_trust_dict

    def getWeight(self, predictor):
        total = 0
        for i in self.set_cover:
            total += self.trust_towards[i]
        if total == 0:
            return 1/len(self.set_cover)
        return self.trust_towards[predictor]/total

    def findWeightedPred(self):
        after_weight_pred = 0
        for predictor in self.current_time_pred.keys():
            weight = self.getWeight(predictor)
            weighted_pred = self.current_time_pred[predictor] * weight
            after_weight_pred += weighted_pred
        return after_weight_pred

    def updatePredictors(self, set_cover):
        for p in set_cover:
            self.sim_total_good_preds[p] += 1
            self.sim_total_pred_from[p] += 1
            self.trust_towards[p] = (self.sim_total_good_preds[p] + self.total_good_preds[p]) / (self.sim_total_pred_from[p] + self.total_pred_from[p])

    def updateBadPredictors(self, set_cover):
        for p in set_cover:
            self.sim_total_pred_from[p] += 1
            self.trust_towards[p] = (self.sim_total_good_preds[p] + self.total_good_preds[p]) / (self.sim_total_pred_from[p] + self.total_pred_from[p])



    def __repr__(self):
        return "Agent({}, {})".format(self.id, self.trust_towards)
