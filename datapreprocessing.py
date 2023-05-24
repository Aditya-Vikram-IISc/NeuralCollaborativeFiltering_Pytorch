import pandas as pd
import argparse

class Datapreprocessor:
    '''
    Args: 
    file_path - path of the location where raw data is stored
    train_path - path to store train data
    test_path - path to store test data

    Returns: train and test files. For each user, his/her latest interaction to be kept in test as per Leave one-out evaluation
    '''
    def __init__(self, file_path = "data/ratings.dat", col_names = ['user_id', 'movie_id', 'rating', 'timestamp']):
        # read the original data file
        self.df = pd.read_table(file_path, sep='::', header=None, names=col_names, engine='python')
        self.col_names = col_names
        
    def train_test_split(self, train_path = "data/ratings_train", test_path= "data/ratings_test"):
        # sort the data at user+timestamp level. Most recent user-item interaction to be kept in test
        # create a new column names rank
        self.df["rank"] = self.df.groupby("user_id")["timestamp"].rank(method = "first", ascending = False)
        
        # all rank 1 to be assigned to test, remaining to train
        self.test = self.df[self.df["rank"] == 1]
        self.train = self.df[~self.df.index.isin(self.test.index)]
        
        # shuffle the train and test df and return the data
        self.train = self.train.sample(frac=1).reset_index(drop=True)
        self.test = self.test.sample(frac=1).reset_index(drop=True)
        
        # save the data in the respective paths
        try:
            self.train[self.col_names].to_csv(train_path, header = False, index = False, sep = '\t')
            self.test[self.col_names].to_csv(test_path, header = False, index = False, sep = '\t')
            print("Train and test data saved successfully!")
        except:
            print("Data request couldn't be processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add arguments to the parser
    parser.add_argument("--file_path", type = str)
    parser.add_argument("--train_path", type = str)
    parser.add_argument("--test_path", type = str)
    args = parser.parse_args()

    # create an instance of the dataset
    datapreprocessor = Datapreprocessor(file_path = args.file_path)
    datapreprocessor.train_test_split(args.train_path, args.test_path)