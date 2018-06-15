class LRConfig(object):
    """
    Holds logistic regression model hyperparams.

    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :param batch_size: batch size for training
    :type batch_size: int
    :param epochs: number of epochs
    :type epochs: int
    :param save_step: when step % save_step == 0, the model
                      parameters are saved.
    :type save_step: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float
    """
    def __init__(self,
                 height=45,
                 width=80,
                 channels=3,
                 classes=3,
                 batch_size=32,
                 epochs=3,
                 save_step=100,
                 learning_rate=0.01,
                 momentum=0.1):
        self.height = height
        self.width = width
        self.classes = classes
        self.channels = channels
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_step = save_step
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __str__(self):
        """
        Get all attributs values.
        :return: all hyperparams as a string
        :rtype: str
        """
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "classes = {}\n".format(self.classes)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "save_step = {}\n".format(self.save_step)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "momentum = {}\n".format(self.momentum)
        return status


class DFNConfig(LRConfig):
    """
    Holds DFN model hyperparams.

    :param architecture: network dense architecture
    :type architecture: list of int
    """
    def __init__(self, architecture=[100, 3]):
        super(DFNConfig, self).__init__()
        self.architecture = architecture

    def __str__(self):
        """
        Get all attributs values.

        :return: all hyperparams as a string
        :rtype: str
        """
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "architecture = {}\n".format(self.architecture)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "save_step = {}\n".format(self.save_step)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "momentum = {}\n".format(self.momentum)
        return status
