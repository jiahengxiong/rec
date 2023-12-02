import numpy as np
import torch
import time

training_user_set, training_item_set, training_set_count = np.load('D:\\rec\\dataset\\datanpy\\data.npy',
                                                                   allow_pickle=True)
num_item = 22348
num_user = 13024

urm = np.zeros((num_user, num_item))

for user in training_user_set:
    items = training_user_set[user]
    for i in items:
        urm[user - 1, i - 1] = 1.0

uum = np.zeros((num_user, num_user))
urm_gpu = torch.from_numpy(urm).cuda()
uum_gpu = torch.from_numpy(uum).cuda()
print(torch.cuda.is_available())
for i in range(num_user):
    start = time.time()
    for j in range(num_user):
        if i != j and torch.max(urm_gpu[i, :]) != 0 and torch.max(urm_gpu[j, :]) != 0:
            uum_gpu[i, j] = torch.sum(torch.mul(urm_gpu[i, :], urm_gpu[j, :])) / (
                        torch.sqrt(torch.sum(urm_gpu[i, :]) * torch.sum(urm_gpu[j, :])) + 100)
    end = time.time()
    print(end - start)

torch.save(uum_gpu, 'uum.pth')