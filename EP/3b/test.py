import os
import unittest
from unittest.util import safe_repr
import numpy as np
import time
from torch.utils.data import TensorDataset
import torch
from util import randomize_in_place
from config import LRConfig, DFNConfig
from DataHolder import DataHolder
from lr_dfn import train_model_img_classification, LogisticRegression, DFN


def run_test(testClass):
    """
    Function to run all the tests from a class of tests.
    :param testClass: class for testing
    :type testClass: unittest.TesCase
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestEP3b(unittest.TestCase):
    """
    Class that test the functions from basic_functions module
    """
    @classmethod
    def setUpClass(cls):
        raw_X = np.load("self_driving_pi_car_data/train_data.npy")
        raw_y = np.load("self_driving_pi_car_data/train_labels.npy")
        randomize_in_place(raw_X, raw_y)
        valid_X = raw_X[0:1000]
        valid_y = raw_y[0:1000]
        test_X = raw_X[1000:2000]
        test_y = raw_y[1000:2000]
        train_X = raw_X[2000:]
        train_y = raw_y[2000:]
        del raw_X
        del raw_y
        cls.command2int = {"forward": 0, "left": 1, "right": 2}
        cls.int2command = {i[1]: i[0] for i in cls.command2int.items()}
        train_dataset = TensorDataset(torch.Tensor(train_X),
                                      torch.Tensor(train_y).type(torch.LongTensor))
        valid_dataset = TensorDataset(torch.Tensor(valid_X),
                                      torch.Tensor(valid_y).type(torch.LongTensor))
        test_dataset = TensorDataset(torch.Tensor(test_X),
                                     torch.Tensor(test_y).type(torch.LongTensor))
        cls.lr_config = LRConfig()
        cls.dfn_config = DFNConfig()
        cls.self_driving_data = DataHolder(cls.lr_config,
                                           train_dataset,
                                           valid_dataset,
                                           test_dataset)
        batch_X, batch_y = next(iter(cls.self_driving_data.train_loader))
        cls.batch_X = batch_X / 255
        cls.total_score = 0

    @classmethod
    def tearDown(cls):
        if os.path.exists("my_test.pkl"):
            os.remove("my_test.pkl")

    def assertTrue(self, expr, msg=None, score=0):
        """Check that the expression is true."""
        if not expr:
            msg = self._formatMessage(msg, "%s is not true" % safe_repr(expr)) # noqa
            raise self.failureException(msg)
        else:
            TestEP3b.total_score += score

    def param_checker(self, config, model):
        input_shape = config.height * config.width * config.channels
        all_params = list(model.parameters())
        msg = "Modelo sem nenhum parâmetro"
        assert all_params != [], msg
        architecture = [input_shape] + config.architecture
        count = 0
        test_shape_W = True
        test_shape_b = True
        for params in all_params:
            shape = tuple(params.shape)
            if len(shape) == 2:
                msgW = "{} != torch.Size([{}, {}])".format(params.shape,
                                                           shape[0],
                                                           shape[1])
                test_shape_W = test_shape_W and shape == (architecture[count + 1], architecture[count])
                if not test_shape_W:
                    print("Erro:", msgW)
            else:
                msgb = "{} != torch.Size([{}])".format(params.shape, shape[0])
                test_shape_b =  shape == (architecture[count + 1],)
                if not test_shape_b:
                    print("Erro:", msgb)
                count += 1
        return test_shape_W and test_shape_b

    def get_mean_acc(self, config_class, model_class, k=7):
        accuracy_list = []
        for i in range(k):
            my_config = config_class()
            my_config.epochs = 1
            my_model = model_class(my_config)
            train_model_img_classification(my_model,
                                           my_config,
                                           self.self_driving_data,
                                           'my_test.pkl',
                                           verbose=False)
            img, labels = next(iter(self.self_driving_data.test_loader))
            img = img / 255
            pred = my_model.predict(img)
            pred = pred.numpy()
            accuracy = np.sum(pred == labels.numpy()) / labels.shape[0]
            accuracy_list.append(accuracy)
        return np.mean(accuracy_list)

    def test_exercise_4(self):
        try:
            lr_model = LogisticRegression(self.lr_config)
            out = lr_model(self.batch_X)
            type_test = out.type() == 'torch.FloatTensor'
            shape_test = out.shape == torch.Size([self.lr_config.batch_size,
                                                  self.lr_config.classes])
            self.assertTrue(type_test,
                            score=0.25,
                            msg="problemas com o tipo da saida do método forward")
            self.assertTrue(shape_test,
                            score=0.25,
                            msg="problemas com o shape da saida do método forward")
            prediction = lr_model.predict(self.batch_X)
            type_test_pred = prediction.type() == 'torch.LongTensor'
            shape_test_pred = prediction.shape == torch.Size([self.lr_config.batch_size])
            self.assertTrue(type_test_pred,
                            score=0.25,
                            msg="problemas com o tipo da saida do método predict")
            self.assertTrue(shape_test_pred,
                            score=0.25,
                            msg="problemas com o shape da saida do método predict")

        except NotImplementedError:
            self.fail('Exercício 4) Falta fazer!')

    def test_exercise_5(self):
        try:
            mean_acc = self.get_mean_acc(LRConfig, LogisticRegression)
            self.assertTrue(os.path.exists("my_test.pkl"),
                            score=0.0,
                            msg="Problemas ao salvar o modelo")
            accuracy_test = mean_acc >= 0.6
            self.assertTrue(accuracy_test,
                            score=2.0,
                            msg="A acurácia média tem que ser maior que 60%")

        except NotImplementedError:
            self.fail('Exercício 5) Falta fazer!')

    def test_exercise_6_train(self):
        try:
            dfn_model = DFN(self.dfn_config)
            out = dfn_model(self.batch_X)
            type_test = out.type() == 'torch.FloatTensor'
            shape_test = out.shape == torch.Size([self.lr_config.batch_size,
                                                  self.lr_config.classes])
            self.assertTrue(type_test,
                            score=0.0,
                            msg="problemas com o tipo da saida do método forward")
            self.assertTrue(shape_test,
                            score=0.0,
                            msg="problemas com o shape da saida do método forward")
            prediction = dfn_model.predict(self.batch_X)
            type_test_pred = prediction.type() == 'torch.LongTensor'
            shape_test_pred = prediction.shape == torch.Size([self.lr_config.batch_size])
            self.assertTrue(type_test_pred,
                            score=0.0,
                            msg="problemas com o tipo da saida do método predict")
            self.assertTrue(shape_test_pred,
                            score=0.0,
                            msg="problemas com o shape da saida do método predict")
            mean_acc = self.get_mean_acc(DFNConfig, DFN, k=3)
            self.assertTrue(os.path.exists("my_test.pkl"),
                            score=0.0,
                            msg="Problemas ao salvar o modelo")
            accuracy_test = mean_acc >= 0.6
            self.assertTrue(accuracy_test,
                            score=1.0,
                            msg="A acurácia média tem que ser maior que 60%")
        except NotImplementedError:
            self.fail('Exercício 6) Falta fazer!')

    def test_exercise_6_arch(self):
        try:
            deep_config1 = DFNConfig(architecture=[400, 300, 200, 100, 50, 10])
            deep_model1 = DFN(deep_config1)
            deep_config2 = DFNConfig(architecture=[800,
                                                   600,
                                                   400,
                                                   300,
                                                   200,
                                                   100,
                                                   50,
                                                   27])
            deep_model2 = DFN(deep_config2)
            deep_config3 = DFNConfig(architecture=[900,
                                                   400,
                                                   300,
                                                   200,
                                                   100,
                                                   50,
                                                   13])
            deep_model3 = DFN(deep_config3)
            shallow_config = DFNConfig(architecture=[10])
            shallow_model = DFN(shallow_config)

            arch1_test = self.param_checker(deep_config1, deep_model1)
            arch2_test = self.param_checker(deep_config2, deep_model2)
            arch3_test = self.param_checker(deep_config3, deep_model3)
            arch4_test = self.param_checker(shallow_config, shallow_model)
            self.assertTrue(arch1_test,
                            score=0.25,
                            msg="erro com architecture=[400, 300, 200, 100, 50, 10]")
            self.assertTrue(arch2_test,
                            score=0.25,
                            msg="erro com architecture=[800, 600, 400, 300, 200, 100, 50, 27]")
            self.assertTrue(arch3_test,
                            score=0.25,
                            msg="erro com architecture=[900, 400, 300, 200, 100, 50, 13]")
            self.assertTrue(arch4_test,
                            score=0.25,
                            msg="erro com architecture=[10]")
        except NotImplementedError:
            self.fail('Exercício 6) Falta fazer!')


if __name__ == '__main__':
    run_test(TestEP3b)
    time.sleep(0.1)
    total_score = TestEP3b.total_score
    print("\nEP3b total_score = ({:.1f} / 5,0)".format(total_score))
    print("\n0.8 pontos restantantes vão ser dados pela competição no kaggle") # noqa

