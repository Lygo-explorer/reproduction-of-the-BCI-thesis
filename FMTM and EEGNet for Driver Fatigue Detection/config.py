# data parameters
electrode_num = 30 # the number of electrodes
sampling_rate = 128 # the sampling rate
time_num = 384 # the number of data in time dimension
batch_size = 64 # batch size, 64 recommended
n_class = 2 # number of classes
partitions = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]] # the dividing rule of electrodes
test_data_rate = 0.3 # the test data rate when splitting the training and testing data

# generic model's parameters
dropout_rate = 0.5 # the dropout rate in the MLP

# FMTM model's parameters
kernel_num = 8 # the number of conv kernel, 64 recommended
feature_num = 29 # the F dimension of input data
alpha_list = [0.5, 0.25, 0.125] # should be a list with 3 elements, the alpha will be used to calculate the conv kernel size
graph_hidden_size = 64 # hidden dimension of graph convolution layer, 256 recommended
attention_hidden_size = 64 # hidden dimension of self-attention layer, 256 recommended

# EEGNet model's parameters
temporal_kernel_num = 8 # the number of temporal conv layer's kernel
spatial_kernel_num = 2 # the number of spatial depthwise conv layer's kernel
separable_kernel_num = 16 # the number of separable conv layer's kernel

# train parameters
device = 'cuda'
learning_rate = 1e-4 # the learning rate while training the model
train_epoch = 100 # the number of training epoch, 50 recommended
random_state = 42 # the random state to keep the experiment repeatable