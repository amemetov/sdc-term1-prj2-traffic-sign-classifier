from abc import ABC, abstractmethod

import numpy as np

from LeNet import LeNetConfig, LeNetParameters, LeNet, LeNetSolver

class RandMethod(ABC):
    @abstractmethod
    def gen_val(self):
        pass

class RandInt(RandMethod):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def gen_val(self):
        return np.random.randint(self.start, self.end)

class RandFloat(RandMethod):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.range = end - start

    def gen_val(self):
        r = np.random.rand()
        # r is from [0, 1), map to [start, end)
        return self.start + r*self.range



class RandomSearch(object):
    def __init__(self, param_distribs, num_iter_search):
        self.param_distribs = param_distribs
        self.num_iter_search = num_iter_search


    def search(self, num_classes, X_train, y_train, X_valid, y_valid):
        input_dim = X_train.shape[1:]

        result_best_history = None
        result_best_valid_loss = None
        result_best_valid_accuracy = 0
        result_best_valid_params = None
        result_best_hyperparams = None

        for i in range(self.num_iter_search):
            params = self.gen_params()
            print("====================================================================")
            print("===== Searching iter:{0}\n hyperparams:{1} =====\n".format(i, params))

            (history, best_valid_loss, best_valid_accuracy, best_valid_params) = self._train(input_dim, num_classes, params, X_train, y_train, X_valid, y_valid)

            if result_best_valid_accuracy < best_valid_accuracy:
                result_best_history = history
                result_best_valid_loss = best_valid_loss
                result_best_valid_accuracy = best_valid_accuracy
                result_best_valid_params = best_valid_params
                result_best_hyperparams = params

        return (result_best_hyperparams, result_best_history, result_best_valid_loss, result_best_valid_accuracy, result_best_valid_params)


    def gen_params(self):
        params = dict()
        for k, v in self.param_distribs.items():
            if isinstance(v, RandMethod):
                params[k] = v.gen_val()
            else:
                params[k] = v[np.random.randint(0, len(v))]

        return params

    def _train(self, input_dim, num_classes, params, X_train, y_train, X_valid, y_valid):
        lenet_config = LeNetConfig(input_dim, num_classes, params)
        lenet_params = LeNetParameters(lenet_config)
        lenet = LeNet(lenet_params)
        solver = LeNetSolver(lenet, X_train, y_train, X_valid, y_valid, debug=False)
        return solver.train()