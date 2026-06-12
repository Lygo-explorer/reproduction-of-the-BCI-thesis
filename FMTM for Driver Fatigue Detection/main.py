from scipy.io import loadmat

# Load the row data
data = loadmat('data/dataset.mat')
unbalanced_data = loadmat('data/unbalanced_dataset.mat')

# check the basic information about the data
print("data")
for k, v in data.items():
    if not k.startswith('__'):
        print(k, type(v), v.shape)

# check the basic information about the unbalanced data
print("unbalanced_data")
for k, v in unbalanced_data.items():
    if not k.startswith('__'):
        print(k, type(v), v.shape)

