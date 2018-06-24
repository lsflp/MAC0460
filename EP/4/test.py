import os
import unittest
from unittest.util import safe_repr
import numpy as np
import time
from torch.utils.data import TensorDataset
import torch
from util import randomize_in_place
from config import CNNConfig
from DataHolder import DataHolder
from cnn import train_model_img_classification, CNN


def run_test(testClass):
    """
    Function to run all the tests from a class of tests.
    :param testClass: class for testing
    :type testClass: unittest.TesCase
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestEP4(unittest.TestCase):
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
        valid_X = valid_X.reshape((-1, 3, 45, 80))
        test_X = test_X.reshape((-1, 3, 45, 80))
        train_X = train_X.reshape((-1, 3, 45, 80))

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
        cls.cnn_config = CNNConfig()
        cls.self_driving_data = DataHolder(cls.cnn_config,
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
            TestEP4.total_score += score

    def param_checker(self, config, model):
        input_shape = config.height * config.width * config.channels
        all_params = list(model.parameters())
        msg = "Modelo sem nenhum parâmetro"
        assert all_params != [], msg
        conv_architecture = config.conv_architecture
        architecture = config.architecture

        count_b = 0
        conv_count_b = 0

        test_shape_F = True
        test_shape_W = True
        test_shape_b = True

        for params in all_params:
            shape = tuple(params.shape)

            if len(shape) == 4:
                filters = conv_architecture[conv_count_b]
                msgF = "model's param: {} !=  expected: {}".format(params.shape[0], filters) # noqa
                test_shape_F = test_shape_F and filters == params.shape[0]

                if not test_shape_F:
                    print("Erro:", msgF)

            if len(shape) == 2:
                W_first_dim = architecture[count_b]
                msgW = "model's param: {} !=  expected: {}".format(params.shape[0], W_first_dim) # noqa
                test_shape_W = test_shape_W and W_first_dim == params.shape[0]

                if not test_shape_W:
                    print("Erro:", msgW)

            elif len(shape) == 1:
                if conv_count_b < len(conv_architecture):
                    b_dim = conv_architecture[conv_count_b]
                    conv_count_b += 1

                else:
                    b_dim = architecture[count_b]
                    count_b += 1

                msgb = "model's param: {} !=  expected: {}".format(params.shape[0], b_dim) # noqa
                test_shape_b = test_shape_b and b_dim == params.shape[0]
                if not test_shape_b:
                    print("Erro:", msgb)

        return test_shape_F and test_shape_W and test_shape_b

    def get_mean_acc(self, config_class, model_class, k=1):
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

    def test_output_shape_and_type(self):
        try:
            cnn_model = CNN(self.cnn_config)
            out = cnn_model(self.batch_X)
            type_test = out.type() == 'torch.FloatTensor'
            shape_test = out.shape == torch.Size([self.cnn_config.batch_size,
                                                  self.cnn_config.classes])
            self.assertTrue(type_test,
                            score=0.5,
                            msg="problemas com o tipo da saida do método forward") # noqa
            self.assertTrue(shape_test,
                            score=0.5,
                            msg="problemas com o shape da saida do método forward") # noqa
            prediction = cnn_model.predict(self.batch_X)
            type_test_pred = prediction.type() == 'torch.LongTensor'
            shape_test_pred = prediction.shape == torch.Size([self.cnn_config.batch_size]) # noqa
            self.assertTrue(type_test_pred,
                            score=0.5,
                            msg="problemas com o tipo da saida do método predict") # noqa
            self.assertTrue(shape_test_pred,
                            score=0.5,
                            msg="problemas com o shape da saida do método predict") # noqa

        except NotImplementedError:
            self.fail('Exercício Falta fazer!')

    def test_cnn_training(self):
        try:
            mean_acc = self.get_mean_acc(CNNConfig, CNN)
            self.assertTrue(os.path.exists("my_test.pkl"),
                            score=0.0,
                            msg="Problemas ao salvar o modelo")
            accuracy_test = mean_acc >= 0.6
            self.assertTrue(accuracy_test,
                            score=4.0,
                            msg="A acurácia média tem que ser maior que 60%")

        except NotImplementedError:
            self.fail('Exercício Falta fazer!')

    def test_different_architectures(self):
        try:
            deep_config1 = CNNConfig(conv_architecture=[32, 16, 12],
                                     architecture=[400, 300, 200, 100, 50, 10])
            deep_model1 = CNN(deep_config1)

            deep_config2 = CNNConfig(conv_architecture=[16, 12, 10],
                                     architecture=[300, 200, 100, 50, 27])
            deep_model2 = CNN(deep_config2)

            deep_config3 = CNNConfig(conv_architecture=[32, 8],
                                     architecture=[500, 13])
            deep_model3 = CNN(deep_config3)

            shallow_config = CNNConfig(conv_architecture=[32, 12, 8],
                                       architecture=[10])
            shallow_model = CNN(shallow_config)
            arch1_test = self.param_checker(deep_config1, deep_model1)
            arch2_test = self.param_checker(deep_config2, deep_model2)
            arch3_test = self.param_checker(deep_config3, deep_model3)
            arch4_test = self.param_checker(shallow_config, shallow_model)
            self.assertTrue(arch1_test,
                            score=1.0,
                            msg="erro com conv_architecture={} e architecture={}".format(deep_config1.architecture, # noqa

                                                                                         deep_config1.conv_architecture)) # noqa

            self.assertTrue(arch2_test,
                            score=1.0,
                            msg="erro com conv_architecture={} e architecture={}".format(deep_config2.architecture, # noqa

                                                                                         deep_config2.conv_architecture)) # noqa

            self.assertTrue(arch3_test,
                            score=1.0,
                            msg="erro com conv_architecture={} e architecture={}".format(deep_config3.architecture, # noqa

                                                                                         deep_config3.conv_architecture)) # noqa

            self.assertTrue(arch4_test,
                            score=1.0,
                            msg="erro com conv_architecture={} e architecture={}".format(shallow_config.architecture, # noqa

                                                                                         shallow_config.conv_architecture)) # noqa

        except NotImplementedError:
            self.fail('Exercício Falta fazer!')


if __name__ == '__main__':
    run_test(TestEP4)
    time.sleep(0.1)
    total_score = TestEP4.total_score
    print("\nEP3b total_score = ({:.1f} / 10,0)".format(total_score))
