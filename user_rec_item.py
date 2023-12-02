import torch
import numpy as np
import pandas as pd

training_user_set, training_item_set, training_set_count = np.load('D:\\rec\\dataset\\datanpy\\data.npy',
                                                                   allow_pickle=True)
csv_file_path = "D:\\rec\\dataset\\data_target_users_test.csv"
num_item = 22347
num_user = 13024

urm = np.zeros((num_user, num_item))

for user in training_user_set:
    items = training_user_set[user]
    for i in items:
        urm[user - 1, i - 1] = 1.0
uum = torch.load('uum.pth')

uum_cpu = uum.cpu()

urm_cpu = torch.from_numpy(urm).cpu()

values, indices = torch.topk(uum_cpu, k=20, dim=1)

uum_knn_cpu = torch.zeros_like(uum_cpu).scatter(dim=1, index=indices, src=values)

rui_cpu = torch.mm(uum_knn_cpu, urm_cpu)

rui_numpy = rui_cpu.numpy()

column_sums = torch.sum(urm_cpu, dim=0)
values, indices = torch.topk(column_sums, k=10)
indices = indices.numpy()
sorted_indices = np.argsort(rui_numpy, axis=1)[:, ::-1]
df = pd.read_csv(csv_file_path)
df['item_list'] = np.nan

for index, row in df.iterrows():
    rec = ''
    user = int(df.loc[index]['user_id'])
    temp = 0
    if max(urm[user - 1, :]) == 0:
        for j in indices:
            rec = rec + str(j + 1) + " "
    else:
        for i in sorted_indices[user - 1, :]:
            if temp == 10:
                break
            if urm[user - 1, i] == 0:
                rec = rec + str(i + 1) + ' '
                temp = temp + 1
            else:
                continue
    df.at[index, 'item_list'] = rec

df.to_csv("submission.csv", index = False)


