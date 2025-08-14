import pandas as pd

file_path = 'junk/GPT4/DialogCC/sample_master.txt'
df = pd.read_csv(file_path, sep='\t')


row = (df[df['id'] == 'te_wow:1174__u3__s159'])


print(list(row['prefix'])[0])