import torch
import model

x = torch.rand(10, 1, 30, 384)
fmtm_model = model.FMTMModel()
y = fmtm_model(x)
print(y.shape)