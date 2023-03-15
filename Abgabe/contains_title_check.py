import pandas as pd

df = pd.read_csv('clean_data_nn.csv')

query = input('What title do you want to check? ')

for title in df['title']:
    if (query in title) == True:
        print(title)
    else:
        pass
