import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LogisticRegression(nn.Module):
    """
    Logistic regression model.
    
    You may find nn.Linear and nn.Softmax useful here.
    
    :param config: hyper params configuration
    :type config: LRConfig
    """
    def __init__(self, config):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(config.height*config.width*config.channels, config.classes)
        self.logits = None

    def forward(self, x):
        """
        Computes forward pass
        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: logits
        :rtype: torch.FloatTensor(shape=[batch_size, number_of_classes])
        """
        self.logits = self.model(x)
        logits = self.logits
        
        return logits
    
    
    def predict(self, x):
        """
        Computes model's prediction
        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: model's predictions
        :rtype: torch.LongTensor(shape=[batch_size])
        """
        if (self.logits is None):
            self.logits = self.forward(x)
            
        softmax = nn.Softmax(1)(self.logits)
        predictions = torch.argmax(softmax, 1)
        
        return predictions   


class DFN(nn.Module):
    """
    Deep Feedforward Network.

    The method self._modules is useful here.
    The class nn.ReLU() is useful too.

    :param config: hyper params configuration
    :type config: DFNConfig
    """
    def __init__(self, config):
        super(DFN, self).__init__()
        # YOUR CODE HERE:
        raise NotImplementedError("falta completar o método __init__ da classe DFN")
        # END YOUR CODE


    def forward(self, x):
        """
        Computes forward pass

        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: logits
        :rtype: torch.FloatTensor(shape=[batch_size, number_of_classes])
        """
        # YOUR CODE HERE:
        raise NotImplementedError("falta completar o método forward da classe DFN")
        # END YOUR CODE
        return logits

    def predict(self, x):
        """
        Computes model's prediction

        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: model's predictions
        :rtype: torch.LongTensor(shape=[batch_size, number_of_classes])
        """
        # YOUR CODE HERE:
        raise NotImplementedError("falta completar o método predict da classe DFN")     
        # END YOUR CODE
        return predictions


def train_model_img_classification(model,
                                   config,
                                   dataholder,
                                   model_path,
                                   verbose=True):
    """
    Train a model for image classification

    :param model: image classification model
    :type model: LogisticRegression or DFN
    :param config: image classification model
    :type config: LogisticRegression or DFN
    :param dataholder: data
    :type dataholder: DataHolder
    :param model_path: path to save model params
    :type model_path: str
    :param verbose: param to control print
    :type verbose: bool
    """
    train_loader = dataholder.train_loader
    valid_loader = dataholder.valid_loader

    best_valid_loss = float("inf")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    
    train_loss = []
    valid_loss = []
    for epoch in range(config.epochs):
        for step, (images, labels) in enumerate(train_loader):
            
            optimizer.zero_grad()
            logits = model.forward(images)
            loss = criterion(logits, labels)
            
            if step % config.save_step == 0:
                v_images, v_labels = next(iter(valid_loader))
                predictions = model.forward(v_images)
                v_loss = criterion(predictions, v_labels)
                
                valid_loss.append(float(v_loss))
                train_loss.append(float(loss))
                if float(v_loss) < best_valid_loss:
                    msg = "\ntrain_loss = {:.3f} | valid_loss = {:.3f}".format(float(loss),float(v_loss))
                    torch.save(model.state_dict(), model_path)
                    best_valid_loss = float(v_loss)
                    if verbose:
                        print(msg, end="")
            
            loss.backward()
            optimizer.step()

    if verbose:
        x = np.arange(1, len(train_loss) + 1, 1)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(x, train_loss, label='train loss')
        ax.plot(x, valid_loss, label='valid loss')
        ax.legend()
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('Train and valid loss')
        plt.grid(True)
        plt.show()
