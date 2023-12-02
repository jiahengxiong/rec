import pandas as pd
import csv

csv_file_path = "D:\\rec\\dataset\\data_train.csv"
txt_file_path = "D:\\rec\\dataset\\data_train.txt"

with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    pre_user = 0
    txt = ''
    for row in csv_reader:
        if row[0] == pre_user:
            txt = txt + ' ' + str(row[1])
        else:
            txt = txt + '\n' + str(row[0]) + ' ' + str(row[1])
        pre_user = row[0]

with open(txt_file_path, 'w') as file:
    file.write(txt)
