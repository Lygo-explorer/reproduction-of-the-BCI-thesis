kernel_num = 10
sampling_rate = 128
alpha_list = [0.5, 0.25, 0.125]
batch_size = 10
channel_num = 30
feature_num = 29
hidden_size = 10
n_class = 2
"""
:param kernel_num: the number of conv kernel
:param sampling_rate: the sampling rate
:param alpha_list: should be a list with 3 elements, the alpha will be used to calculate the conv kernel size
:param batch_size: batch size
:param channel_num: the number of channels
:param feature_num: the F dimension of input data
:param hidden_size: hidden dimension of graph convolution layer
:param n_class: number of classes
"""