import pandas as pd


class dataprep:
    def __init__(self, file_path, col_names = ['user_id', 'movie_id', 'rating', 'timestamp']):
        # read the original data file
        self.df = pd.read_table(file_path, sep='::', header=None, names=col_names, engine='python')
        
    def train_test_split(self):
        # sort the data at user+timestamp level. Most recent user-item interaction to be kept in test
        # create a new column names rank
        self.df["rank"] = self.df.groupby("user_id")["timestamp"].rank(method = "first", ascending = False)
        
        # all rank 1 to be assigned to test, remaining to train
        self.test = self.df[self.df["rank"] == 1]
        self.train = self.df[~self.df.index.isin(self.test.index)]
        
        # shuffle the train and test df and return the data
        self.train = self.train.sample(frac=1).reset_index(drop=True)
        self.test = self.test.sample(frac=1).reset_index(drop=True)
        
        return self.train[["user_id", "movie_id"]], self.test[["user_id", "movie_id"]]