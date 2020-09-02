import pandas as pd

train_df = pd.read_csv(r"D:\Users\HBZ\PycharmProjects\pythonProject2\yelp_review_polarity_csv\train.csv", header=None)
train_df.head()

test_df = pd.read_csv(r"D:\Users\HBZ\PycharmProjects\pythonProject2\yelp_review_polarity_csv\test.csv", header=None)
test_df.head()

train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)

train_df.head()

test_df.head()

train_df_bert = pd.DataFrame({
    'id':range(len(train_df)),
    'label':train_df[0],
    'alpha':['a']*train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
})

train_df_bert.head()

dev_df_bert = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

dev_df_bert.head()

train_df_bert.to_csv(r"D:\Users\HBZ\PycharmProjects\pythonProject2\yelp_review_polarity_csv\train.csv",
                     sep='\t', index=False, header=False)

dev_df_bert.to_csv(r"D:\Users\HBZ\PycharmProjects\pythonProject2\yelp_review_polarity_csv\test.csv",
                   sep='\t', index=False, header=False)
