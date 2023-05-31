import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

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
        self.n_users = self.df["user_id"].nunique()
        self.n_movies = self.df["movie_id"].nunique()

    def get_user_and_movie_counts(self):
        return self.n_users, self.n_movies
        
    def train_test_split(self, trainpos_path = "data/ratings_train_pos", testpos_path= "data/ratings_test_pos", \
                        trainneg_path = "data/ratings_train_neg", testneg_path= "data/ratings_test_neg", \
                        n_train_negatives =4, n_test_negatives=100):
        # sort the data at user+timestamp level. Most recent user-item interaction to be kept in test
        # create a new column names rank
        self.df["rank"] = self.df.groupby("user_id")["timestamp"].rank(method = "first", ascending = False)
        
        # all rank 1 to be assigned to test, remaining to train
        self.test = self.df[self.df["rank"] == 1]
        self.train = self.df[~self.df.index.isin(self.test.index)]
        
        # shuffle the train and test df and return the data
        self.train = self.train[['user_id', 'movie_id']].sample(frac=1).reset_index(drop=True)
        self.test = self.test[['user_id', 'movie_id']].sample(frac=1).reset_index(drop=True)
        
        # get the negative data for testing i.e. for each user id get 100 movieids that he/she hasn't interacted
        train_list = list(zip(self.train["user_id"], self.train["movie_id"]))
        test_list = list(zip(self.test["user_id"], self.test["movie_id"]))

        try:
            # save the list as csv at the respective paths
            np.savetxt(trainpos_path, train_list, delimiter ="\t", fmt ='% s')
            print("Train positive data created!")
        except:
            print("Train positive data request couldn't be processed!")

        try:
            np.savetxt(testpos_path, test_list, delimiter ="\t", fmt ='% s')
            print("Test positive data created!")
        except:
            print("Test positive data request couldn't be processed!")

        # create a list containing (user_id, movie_id) for training. To be treated as negatives. 4 for every positive instance
        print("Train negative data creation started!")
        train_neg_list = []
        for usr_idx, mov_idx in tqdm(train_list):
            for i in range(n_train_negatives):
                # get a movie id
                id_neg = np.random.randint(1, self.n_movies + 1)
                while (usr_idx, id_neg) in train_list or (usr_idx, id_neg) in test_list:
                    id_neg = np.random.randint(1, self.n_movies + 1)
                train_neg_list.append((usr_idx, id_neg))

        try:
            # save the list as csv at the respective paths
            np.savetxt(trainneg_path, train_neg_list, delimiter ="\t", fmt ='% s')
            print("Train negative data created!")
        except:
            print("Train negative data request couldn't be processed!")


        # create a list containing (user_id, movie_id) for testing. To be treated as negatives
        print("Test negative data creation started!")
        test_neg_list = []
        for usr_idx, mov_idx in tqdm(test_list):
            for i in range(n_test_negatives):
                # get an movie id not present in train or test
                id_neg = np.random.randint(1, self.n_movies + 1)
                while ((usr_idx, id_neg) in train_list) or ((usr_idx, id_neg) in train_neg_list) or (id_neg == mov_idx):
                    id_neg = np.random.randint(1, self.n_movies + 1)
                test_neg_list.append((usr_idx, id_neg))
        
        # save the data in the respective paths
        try:
            # save the list as csv at the respective paths
            np.savetxt(testneg_path, test_neg_list, delimiter ="\t", fmt ='% s')
            print("Test negative data created!")
        except:
            print("negative data request couldn't be processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add arguments to the parser
    parser.add_argument("--file_path", type = str, default = "data/ratings.dat")
    parser.add_argument("--trainpos_path", type = str, default = "data/ratings_train_pos")
    parser.add_argument("--testpos_path", type = str, default = "data/ratings_test_pos")
    parser.add_argument("--trainneg_path", type = str, default = "data/ratings_train_neg")
    parser.add_argument("--testneg_path", type = str, default = "data/ratings_test_neg")
    parser.add_argument("--n_train_negatives", type = int, default = 4)
    parser.add_argument("--n_test_negatives", type = int, default = 100)
    args = parser.parse_args()

    # create an instance of the dataset
    datapreprocessor = Datapreprocessor(file_path = args.file_path)
    n_users, n_movies = datapreprocessor.get_user_and_movie_counts()
    print("Dataset preparation started! ")
    datapreprocessor.train_test_split(trainpos_path = args.trainpos_path, testpos_path = args.testpos_path, \
                                      trainneg_path = args.trainneg_path, testneg_path = args.testneg_path, \
                                      n_train_negatives = args.n_train_negatives, n_test_negatives = args.n_test_negatives)
    print(f"Dataset created for num_users : {n_users}, and num_movies : {n_movies} ")