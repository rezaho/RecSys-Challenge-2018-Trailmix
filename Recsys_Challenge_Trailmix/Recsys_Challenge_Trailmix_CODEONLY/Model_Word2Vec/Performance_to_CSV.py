import pandas as pd

result = []
with open('Model_2_Performance_500.txt', 'r') as f:
    for line in f:
        row = line[:-1].split(' ')
        result.append([round(float(row[2]),4),round(float(row[4]),4)])

result_new = []
result_new.append(['','R_pre(mean)', 'R_pre(std)','NDCG(mean)', 'NDCG(std)','Clicks(mean)', 'Clicks(std)'])
for i in range (10):
    result_new.append(['Task ' + str(i)] + result[3*i+0] + result[3*i+1] + result[3*i+2])

a = pd.DataFrame(result_new)

a.to_csv('Model_2_Performance_500_CSV.csv')
