import utils
import torch
import config
from torch.utils.data import Dataset, DataLoader
from model.EEGNet import EEGNet
from model.Conformer import Conformer
from model.FBCNet import FBCNet
from model.ATCNet import ATCNet
from matplotlib import pyplot as plt
import torchmetrics
from pathlib import Path
import numpy as np

# Set the model, optimize configs and plot saving path
model_name = config.model_name
data_name = config.data_name
plot_path = Path('figures')
plot_path = plot_path/data_name/model_name/'experiment'
plot_path = utils.mkdir_with_suffix(plot_path)
log_path = f"logs/{data_name}/{model_name} train.log"

# open the log
with open(log_path, "a") as f:
    f.write("============================================\n")
    f.write(f"experiment id: {plot_path.stem}\n")
    f.write("----------configuration----------\n")
    f.write(f"batch size: {config.batch_size}\n")
    f.write(f"learning rate: {config.learning_rate}\n")
    f.write(f"training epochs: {config.train_epoch}\n")
    f.write(f"beta1, beta2: {config.beta1}, {config.beta2}\n")
    f.write(f"weight decay: {config.weight_decay}\n")
    f.write("----------train process----------\n")

# Set the manual seed
torch.manual_seed(config.random_state)

# the dataset customized class
class BCIIVDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# load the data
if data_name == 'BCICIV_2a_gdf':
    train_datas, test_datas, train_labels, test_labels = utils.load_bciciv2a("data/BCICIV_2a_gdf",
                                                                             3.0, 6.996)  # choose the 3-7 second's data
elif data_name == 'BCICIV_2b_gdf':
    train_datas, test_datas, train_labels, test_labels = utils.load_bciciv2b("data/BCICIV_2b_gdf",
                                                                            3.0, 6.996) # choose the 3-7 second's data
else:
    raise ValueError("Unknown dataset")

# get the basic information of data
_, C, T = train_datas[0].shape
n_class = np.max(train_labels[0]).item()+1
print(train_datas[0].shape)
print(train_labels[0].shape)
print(test_datas[0].shape)
print(test_labels[0].shape)

# loop through all the subjects
total_max_test_acc = 0
for i in range(len(train_datas)):
    print(f"dealing with subject {i}")
    with open(log_path, "a") as f:
        f.write(f"subject {i}:\n")

    # get the i-th subject's data
    train_data, train_label = train_datas[i], train_labels[i]
    test_data, test_label = test_datas[i], test_labels[i]

    if model_name == 'Conformer':

        # Use a band pass to deal with data noise
        train_data = utils.cheby1_bandpass(train_data, 4, 40, config.sampling_rate)
        test_data = utils.cheby1_bandpass(test_data, 4, 40, config.sampling_rate)

    # train data augmentation
    train_data, train_label = utils.segmentation_reconstruction(train_data, train_label)

    # the FBCNet should apply the cheby1_bandpass to the data
    if model_name=='FBCNet':
        train_data_list = []
        test_data_list = []
        for band_idx in range(9):
            lowcut = band_idx*4+4
            highcut = lowcut+4
            train_data_list.append(utils.cheby1_bandpass(train_data, lowcut, highcut, config.sampling_rate))
            test_data_list.append(utils.cheby1_bandpass(test_data, lowcut, highcut, config.sampling_rate))
        train_data = np.stack(train_data_list).transpose(1, 0, 2, 3)
        test_data = np.stack(test_data_list).transpose(1, 0, 2, 3)

    # Turn the data to tensor and reshape it to proper size
    train_data = torch.from_numpy(train_data)
    train_data = train_data.type(torch.float)
    train_label = torch.from_numpy(train_label)
    train_label = train_label.type(torch.long)
    test_data = torch.from_numpy(test_data)
    test_data = test_data.type(torch.float)
    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.long)
    print("train data shape: ", train_data.shape, "train label shape: ", train_label.shape)
    print("test data shape: ", test_data.shape, "test label shape: ", test_label.shape)

    # data standardize
    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean)/std
    test_data  = (test_data - mean)/std

    # create instance of datasets
    train_dataset = BCIIVDataset(train_data, train_label)
    test_dataset = BCIIVDataset(test_data, test_label)

    # Create a dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Set up the device
    device = config.device

    # Create an instance of our model
    if model_name == 'EEGNet':
        model_0 = EEGNet(C, n_class)
    elif model_name == 'Conformer':
        model_0 = Conformer(C, n_class, T)
    elif model_name == 'FBCNet':
        model_0 = FBCNet(C, T, n_class)
    elif model_name == 'ATCNet':
        model_0 = ATCNet(C, n_class)
    else:
        raise ValueError("Unknown model")
    if i==0:
        print("model structure")
        print(model_0)

    # put the model to device
    model_0.to(device)

    # Set up the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set up the accuracy function
    if data_name == 'BCICIV_2b_gdf':
        acc_fn = torchmetrics.Accuracy(task="binary")
        acc_fn.to(device)
    elif data_name == 'BCICIV_2a_gdf':
        acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)
        acc_fn.to(device)
    else:
        raise ValueError("Unknown dataset")

    # Set up the optimizer
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model_0.parameters(),
                                 betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

    # Set up the recording arrays and variables
    epoch_values = []
    loss_values = []
    test_loss_values = []
    acc_values = []
    test_acc_values = []
    max_test_acc = 0
    loss_max_test_acc = 0
    epoch_max_acc = 0

    # training and testing loop
    epochs = config.train_epoch
    for epoch in range(epochs):
        epoch_values.append(epoch)

        # turn the model to training mode
        model_0.train()

        # loop through the batches
        total_loss, total_acc = 0, 0
        for x, y in train_dataloader:

            # push the data to device
            x, y = x.to(device), y.to(device)

            # training process
            logits = model_0(x)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            loss = loss_fn(logits, y)
            acc = acc_fn(preds, y)
            total_loss += loss.item()
            total_acc += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # the FBCNet needs to keep the spatial layer's weight's norm smaller than 2
            if model_name == 'FBCNet':
                with torch.no_grad():
                    w = model_0.spatial_conv.weight
                    norm = w.norm(2, dim=(1, 2, 3), keepdim=True)
                    desired = torch.clamp(norm, max=2.0)
                    model_0.spatial_conv.weight *= desired / (1e-8 + norm)

        # recording the avg train indexes
        avg_loss = total_loss / len(train_dataloader)
        avg_acc = total_acc / len(train_dataloader)
        loss_values.append(avg_loss)
        acc_values.append(avg_acc)

        # turn the model to evaluating mode
        model_0.eval()
        with torch.inference_mode():
            total_loss, total_acc = 0, 0
            for x_test, y_test in test_dataloader:

                # push the data to device
                x_test, y_test = x_test.to(device), y_test.to(device)

                # testing process
                logits_test = model_0(x_test)
                preds_test = torch.argmax(torch.softmax(logits_test, dim=1), dim=1)
                loss_test = loss_fn(logits_test, y_test)
                acc_test = acc_fn(preds_test, y_test)
                total_loss += loss_test.item()
                total_acc += acc_test.item()

            # recording the avg test indexes
            avg_test_loss = total_loss / len(test_dataloader)
            avg_test_acc = total_acc / len(test_dataloader)
            test_loss_values.append(avg_test_loss)
            test_acc_values.append(avg_test_acc)

            # updating the max test acc, loss, epoch
            if avg_test_acc > max_test_acc:
                max_test_acc = avg_test_acc
                loss_max_test_acc = avg_test_loss
                epoch_max_acc = epoch

        # show what's happen
        if epoch % 10 == 0 or epoch==epochs-1:
            print(f"Epoch: {epoch} | loss: {avg_loss:.4f}, acc: {avg_acc*100:.2f}% | test loss: {avg_test_loss:.4f} | test acc: {avg_test_acc*100:.2f}%")

    total_max_test_acc += max_test_acc
    with open(log_path, "a") as f:
        f.write(f"max test acc: {max_test_acc*100:.2f}% | Epoch when acc is max: {epoch_max_acc} | test loss when acc is max: {loss_max_test_acc:.4f}\n")

    # plot the training and testing loss in each epoch
    plt.figure()
    plt.plot(epoch_values, loss_values, 'b-', label='loss')
    plt.plot(epoch_values, test_loss_values, 'r-', label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig(f'{plot_path}/{model_name}_subject {i} loss.png')

    # plot the training and testing accuracy in each epoch
    plt.figure()
    plt.plot(epoch_values, acc_values, 'b-', label='accuracy')
    plt.plot(epoch_values, test_acc_values, 'r-', label='test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig(f'{plot_path}/{model_name}_subject {i} accuracy.png')

# write average max test accuracy and the end of the log
with open(log_path, "a") as f:
    f.write("\n")
    f.write(f"avg max test acc: {(total_max_test_acc/9)*100:.2f}%\n")
    f.write("============================================\n")
    f.write("\n")