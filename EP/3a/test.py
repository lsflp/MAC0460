import unittest
from unittest.util import safe_repr
import numpy as np
import time
from util import get_housing_prices_data
from util import add_feature_ones
from basic_functions import graph1, graph2, SGD_with_momentum


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """

    X_T = X.T
    all_mean = []
    all_std = []
    for i in range(X_T.shape[0]):
        all_mean.append(np.mean(X_T[i]))
        all_std.append(np.std(X_T[i]))

    X_out = (X - all_mean) / all_std

    return X_out


def run_test(testClass):
    """
    Function to run all the tests from a class of tests.
    :param testClass: class for testing
    :type testClass: unittest.TesCase
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestEP3a(unittest.TestCase):
    """
    Class that test the functions from basic_functions module
    """
    @classmethod
    def setUpClass(cls):
        X, y = get_housing_prices_data(N=250, verbose=False)
        train_X_norm = standardize(X)
        train_y_norm = standardize(y)
        cls.train_y_norm = train_y_norm.astype("float64")
        cls.train_X_1 = add_feature_ones(train_X_norm)
        cls.total_score = 0

    def assertTrue(self, expr, msg=None, score=0):
        """Check that the expression is true."""
        if not expr:
            msg = self._formatMessage(msg, "%s is not true" % safe_repr(expr)) # noqa
            raise self.failureException(msg)
        else:
            TestEP3a.total_score += score

    def test_exercise_1(self):
        try:
            count1 = True
            count2 = True
            for _ in range(1000):
                a_np = np.random.rand(1)
                b_np = np.random.rand(1)
                c_np = np.random.rand(1)
                f, auto_grad, user_grad = graph1(a_np, b_np, c_np)
                manual_f = (a_np * c_np) / (a_np + b_np)
                test1 = np.isclose(f.data.numpy()[0], manual_f, atol=1e-4)
                test2 = np.isclose(auto_grad.numpy()[0], user_grad)
                count1 = count1 and test1
                count2 = count2 and test2
            if not count1:
                print("Valor do f com problemas")
            if not count2:
                print("Derivada parcial com problemas")
            self.assertTrue(count1 and count2, score=1.4)
        except NotImplementedError:
            self.fail('Exercício 1) Falta fazer!')

    def test_exercise_2(self):
        try:
            count1 = True
            count2 = True
            iterations = 1000
            sizes = np.random.randint(2, 10, size=(iterations))
            for i in range(iterations):
                size = sizes[i]
                W_np = np.random.rand(size, size)
                x_np = np.random.rand(size, 1)
                b_np = np.random.rand(size, 1)
                f, auto_grad, user_grad = graph2(W_np, x_np, b_np)
                manual_f = np.sum(sigmoid(np.matmul(W_np, x_np) + b_np))
                test1 = np.isclose(f.data.numpy(), manual_f, atol=1e-4)
                test2 = np.allclose(auto_grad.numpy(), user_grad)
                count1 = count1 and test1
                count2 = count2 and test2
            if not count1:
                print("Valor do f com problemas")
            if not count2:
                print("Gradiente com problemas")
            self.assertTrue(count1 and count2, score=1.4)
        except NotImplementedError:
            self.fail('Exercício 2) Falta fazer!')

    def test_exercise_3(self):
        try:
            init = time.time()
            w = np.array([[9.2], [-30.3]])
            w, weights_history, cost_history = SGD_with_momentum(X=self.train_X_1, # noqa
                                                                 y=self.train_y_norm, # noqa
                                                                 inital_w=w, # noqa
                                                                 iterations=1000, # noqa
                                                                 batch_size=32, # noqa
                                                                 learning_rate=0.01, # noqa
                                                                 momentum=0.01) # noqa
            init = time.time() - init
            self.assertTrue(cost_history[-1] < cost_history[0], score=0.7 / 4) # noqa
            self.assertTrue(type(w) == np.ndarray, score=0.7 / 4)
            self.assertTrue(len(weights_history) == len(cost_history), score=0.7 / 4) # noqa
            self.assertTrue(init < 2.5, score=0.7 / 4)
        except NotImplementedError:
            self.fail('Exercício 3) Falta fazer!')


if __name__ == '__main__':
    run_test(TestEP3a)
    time.sleep(0.1)
    total_score = TestEP3a.total_score
    print("\nEP2 total_score = ({:.1f} / 3.5)".format(total_score))
    print("\n0.7 pontos restantantes vão ser dados pelo teste da função SGD_with_momentum nos dados do kaggle") # noqa
