# data parameters
data_name = 'BCICIV_2a_gdf' # the name of dataset, it should be included in the directories' name under the data folder
sampling_rate = 250 # the sampling rate

# generic model's parameters
model_name = 'ATCNet' # the name of the model, it should be the same as one of the files' names in model folder
dropout_rate = 0.5 # the dropout rate in the MLP

# EEGNet model's parameters
temporal_kernel_num = 16 # the number of temporal conv layer's kernel
spatial_kernel_num = 4 # the number of spatial depthwise conv layer's kernel
separable_kernel_num = 32 # the number of separable conv layer's kernel

# Conformer model's parameters
temporal_conv_out_channel = 40 # the output channel of temporal conv layer, 40 recommended
head_num = 10 # the head number of multi head self-attention, 10 recommended
execution_times = 6 # the number of multi head self-attention layers, 6 recommended
embedding_size = 40 # the embedding size of self-attention input, it should be fully divided by head_num, 40 recommended

# FBCNet model's parameters
the_spatial_kernel_num = 32 # the number of spatial depthwise conv layer's kernel, 32 recommended
variance_window = 250 # the size of window in the temporal variance layer, sampling rate recommended

# ATCNet model's parameters
temporal_conv_kernel_num = 16 # the number of temporal conv layer's kernel, 16 recommended
channel_kernel_num = 2 # the number of channel depthwise conv layer's kernel, 2 recommended
stride_of_avg_pool2 = 7 # the stride of avg pool 2 layer in the T dimension, 7 recommended
n_head = 2 # the head number of multi head self-attention, 2 recommended
n_windows = 5 # the number of windows in convolutional based sliding window, 5 recommended
tcb_residual_num = 2 # the number of residual layer in temporal convolutional block. Setting of this parameter should consider the tcb_kernel_size, let the T dimension's output of tcb is 1. 2 recommended
tcb_kernel_size = 4 # the kernel size in temporal convolutional block. Setting of this parameter should consider the tcb_residual_num, let the T dimension's output of tcb is 1. 4 recommended

# train parameters
device = 'cuda'
batch_size = 64 # batch size, 64 recommended
learning_rate = 2e-4 # the learning rate while training the model
beta1, beta2 = 0.5, 0.999
weight_decay = 1e-4
train_epoch = 50 # the number of training epoch, 50 recommended
random_state = 42 # the random state to keep the experiment repeatable