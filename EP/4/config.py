class CNNConfig(object):
    """
    Holds model hyperparams.
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :param architecture: network dense architecture
    :type architecture: list of int
    :param conv_architecture: convolutional architecture
    :type conv_architecture: list of int
    :param kernel_sizes: filter sizes
    :type kernel_sizes: list of int
    :param pool_kernel: pooling filter sizes
    :type pool_kernel: list of int
    :param batch_size: batch size for training
    :type batch_size: int
    :param epochs: number of epochs
    :type epochs: int
    :param save_step: when step % save_step == 0, the model
                      parameters are saved.
    :type save_step: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    """
    def __init__(self,
                 height=45,
                 width=80,
                 channels=3,
                 classes=3,
                 architecture=[100, 3],
                 conv_architecture=[12, 16],
                 kernel_sizes=None,
                 pool_kernel=None,
                 save_step=100,
                 batch_size=32,
                 epochs=1,
                 learning_rate=0.0054,
                 momentum=0.1):
        if kernel_sizes is None:
            self.kernel_sizes = [5] * len(conv_architecture)
        else:
            self.kernel_sizes = kernel_sizes
        if pool_kernel is None:
            self.pool_kernel = [2] * len(conv_architecture)
        else:
            pool_kernel = self.pool_kernel
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes
        self.architecture = architecture
        self.conv_architecture = conv_architecture
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
        status += "architecture = {}\n".format(self.architecture)
        status += "conv_architecture = {}\n".format(self.conv_architecture)
        status += "kernel_sizes = {}\n".format(self.kernel_sizes)
        status += "pool_kernel = {}\n".format(self.pool_kernel)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "momentum = {}\n".format(self.momentum)
        status += "save_step = {}\n".format(self.save_step)
        
        
        return status
