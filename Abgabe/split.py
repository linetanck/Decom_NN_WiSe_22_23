import pandas as pd

df = pd.read_csv('/data/clean_data_nn.csv')

training = df.sample(frac = 0.7)

df=df.drop(training.index)

validation = df.sample(frac = 0.67)

df=df.drop(validation.index)

test = df

training.to_csv('Documents/leuphana/3/neural_networks/project/data/training_set.csv', index=False)
validation.to_csv('Documents/leuphana/3/neural_networks/project/data/validation_set.csv', index=False)
test.to_csv('Documents/leuphana/3/neural_networks/project/data/test_set.csv', index=False)

