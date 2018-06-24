import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CNN(nn.Module):
    """
    Convolutional neural network model.

    You may find nn.Conv2d, nn.MaxPool2d and self.add_module useful here.

    :param config: hyper params configuration
    :type config: CNNConfig
    """
    def __init__(self,
                 config):
        super(CNN, self).__init__()
        
        layers = list()        
        channels = [config.channels] + config.conv_architecture
        dim = [config.batch_size, config.channels, config.height, config.width]
        
        for i in range(len(config.conv_architecture)):
            layers.append(nn.Conv2d(channels[i], channels[i+1], config.kernel_sizes[i]))
            dim[1] = config.conv_architecture[i] # Trocando o número de canais
            dim[2] -= config.kernel_sizes[i]     # Ajustando a altura
            dim[3] -= config.kernel_sizes[i]     # Ajustando a largura
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(config.pool_kernel[i]))
            dim[2] = dim[2]//config.pool_kernel[i] + (dim[2] % config.pool_kernel[i] > 0) # Redimensionando a altura
            dim[3] = dim[3]//config.pool_kernel[i] + (dim[3] % config.pool_kernel[i] > 0) # Redimensionando a largura
            
        self.conv = nn.Sequential(*layers)
        
        network = list()
        network.append(nn.Linear(dim[1]*dim[2]*dim[3], config.architecture[0]))
        
        for i in range(1, len(config.architecture)):
            network.append(nn.ReLU())
            network.append(nn.Linear(config.architecture[i-1], config.architecture[i]))
            
        self.hidden = nn.Sequential(*network)

    def forward(self, x):
        """
        Computes forward pass

        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: logits
        :rtype: torch.FloatTensor(shape=[batch_size, number_of_classes])
        """
        out = self.conv(x)
        out = out.view(out.shape[0], out.shape[1]*out.shape[2]*out.shape[3])
        logits = self.hidden(out)

        return logits

    def predict(self, x):
        """
        Computes model's prediction

        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: model's predictions
        :rtype: torch.LongTensor(shape=[batch_size])
        """
        softmax = nn.Softmax(1)(self.forward(x))
        predictions = torch.argmax(softmax, 1)

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
            
            model.zero_grad()
            logits = model(images/255)
            loss = criterion(logits.type("torch.FloatTensor"), labels)
            
            if step % config.save_step == 0:
                
                v_images, v_labels = next(iter(valid_loader))
                predictions = model(v_images/255)
                v_loss = criterion(predictions.type("torch.FloatTensor"), v_labels)
                
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