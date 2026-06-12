# data parameters
electrode_num = 30 # the number of electrodes
sampling_rate = 128 # the sampling rate
feature_num = 29 # the F dimension of input data
batch_size = 64 # batch size, 64 recommended
n_class = 2 # number of classes
partitions = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]] # the dividing rule of electrodes
test_data_rate = 0.2 # the test data rate when splitting the training and testing data

# model parameters
kernel_num = 8 # the number of conv kernel, 64 recommended
alpha_list = [0.5, 0.25, 0.125] # should be a list with 3 elements, the alpha will be used to calculate the conv kernel size
graph_hidden_size = 64 # hidden dimension of graph convolution layer, 256 recommended
attention_hidden_size = 64 # hidden dimension of self-attention layer, 256 recommended
dropout_rate = 0.5 # the dropout rate in the MLP

# train parameters
device = 'cuda'
learning_rate = 1e-3 # the learning rate while training the model
train_epoch = 10 # the number of training epoch, 50 recommended
random_state = 42 # the random state to keep the experiment repeatable