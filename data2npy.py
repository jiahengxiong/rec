import os
import pdb
from collections import defaultdict
import numpy as np

training_path = "D:\\rec\\dataset\\data_train.txt"
testing_path = "D:\\rec\\dataset\\data_train.txt"
val_path = "D:\\rec\\dataset\\data_train.txt"

path_save_base = "D:\\rec\\dataset\\datanpy"
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    os.makedirs(path_save_base)

train_data_user = defaultdict(set)
train_data_item = defaultdict(set)
links_file = open(training_path)
num_u = 0
num_u_i = 0
for _, line in enumerate(links_file):
    line = line.strip('\n')
    tmp = line.split(' ')
    num_u_i += len(tmp) - 1
    num_u += 1
    u_id = int(tmp[0])
    for i_id in tmp[1:]:
        train_data_user[u_id].add(int(i_id))
        train_data_item[int(i_id)].add(u_id)
np.save("D:\\rec\\dataset\\datanpy\\data.npy", [train_data_user, train_data_item, num_u_i])
print(num_u, num_u_i)

