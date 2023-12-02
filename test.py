import numpy as np

# 假设 rui_numpy 是你的 NumPy 数组
# 这里用一个简单的示例数组代替，你需要替换成真实的 rui_numpy
rui_numpy = np.array([[0.5, 0.8, 0.2, 0.7],
                      [0.3, 0.6, 0.9, 0.1]])

# 获取每一行按降序排列的索引
sorted_indices = np.argsort(rui_numpy, axis=1)[:, ::-1]

# 获取每一行的前50个元素的索引
top_50_indices = sorted_indices[:, :50]

print(top_50_indices)

"""import csv
import pandas as pd
import numpy as np
csv_file_path = "D:\\rec\\dataset\\data_target_users_test.csv"

df = pd.read_csv(csv_file_path)
df['rec'] = np.nan
df['rec'] = df['rec'].astype(object)
df.at[2,'rec'] = "1 2 3 4 5 6 7 8"
print(df.loc[3])"""

"""import numpy as np
import pandas as pd
training_user_set, training_item_set, training_set_count = np.load('D:\\rec\\dataset\\datanpy\\data.npy',
                                                                   allow_pickle=True)
csv_file_path = "D:\\rec\\dataset\\data_target_users_test.csv"

df = pd.read_csv(csv_file_path)
df['rec'] = np.nan
df['rec'] = df['rec'].astype(object)
df.at[2,'rec'] = "1 2 3 4 5 6 7 8"
for index, row in df.iterrows():
    print(index)"""

