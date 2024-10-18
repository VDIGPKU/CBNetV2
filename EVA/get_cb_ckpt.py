import torch

ckpt=torch.load('eva02_B_pt_in21k_p14to16.pt',map_location=torch.device("cpu"))

model = ckpt['model']

keys = list(model.keys())

for k in keys:
    model['backbone.cb_net.' + k] = model[k]
    model['backbone.net.' + k] = model[k]
    del model[k]

ckpt['model'] = model
torch.save(ckpt, 'cb_eva02_B_pt_in21k_p14to16.pth')
