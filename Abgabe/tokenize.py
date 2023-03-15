import torchtext
from torchtext.data import get_tokenizer
from torchtext.data.utils import ngrams_iterator
import pandas as pd

train = pd.read_csv('data/csv/sf_training_set.csv')
test = pd.read_csv('data/csv/sf_test_set.csv')
validation = pd.read_csv('data/csv/sf_validation_set.csv')

tokenizer = get_tokenizer('basic_english')

def tokenizing(df):
    new_column = []
    for i in df['title']:
        tokens = tokenizer(i)
        new_column.append(i)

    df['tokens'] = new_column

tokenizing(train)
tokenizing(validation)
tokenizing(test)

train.to_csv('data/tokenized/sf_token_training_set.csv', index=False)
validation.to_csv('data/tokenized/sf_token_validation_set.csv', index=False)
test.to_csv('data/tokenized/sf_token_test_set.csv', index=False)



#onehot encoding
#C:\Users\Line\Documents\leuphana\3\neural_networks\project\data\csv