import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import random



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
        train_pos_list = list(zip(self.train["user_id"], self.train["movie_id"]))
        test_pos_list = list(zip(self.test["user_id"], self.test["movie_id"]))

        # save the datasets
        try:
            # save the list as csv at the respective paths
            np.savetxt(trainpos_path, train_pos_list, delimiter ="\t", fmt ='% s')
            print("Train positive data created!")
        except:
            print("Train positive data request couldn't be processed!")

        try:
            np.savetxt(testpos_path, test_pos_list, delimiter ="\t", fmt ='% s')
            print("Test positive data created!")
        except:
            print("Test positive data request couldn't be processed!")

        
        # transform the train_pos_list and test_pos_list to dictionaries
        train_pos_dict = {}
        for usr_id, item_id in train_pos_list:
            try:
                train_pos_dict[usr_id].append(item_id)
            except:
                train_pos_dict[usr_id] = [item_id]

        # create a list containing (user_id, movie_id) for training. To be treated as negatives. 4 for every positive instance
        print("Train & Test negative data creation started!")
        train_neg_list = []
        test_neg_list = []
        for usr_idx, mov_idx in tqdm(test_pos_list):
            # get all movies ids corresponding to the userid
            tr_pos = train_pos_dict[usr_idx]
            all_movies = set(range(1, self.n_movies+1))
            diff_movies = all_movies.difference(set(tr_pos), {mov_idx})
            
            # get 100 negatives for test and 4 per positive instance for train
            n_train = n_train_negatives * len(tr_pos)
            try:
                tr_neg_idx = random.sample(list(diff_movies), n_train)
            except:
                tr_neg_idx = np.random.choice(list(diff_movies), size = n_train, replace = True)
            try:
                te_neg_idx = random.sample(list(diff_movies.difference(tr_neg_idx)), n_test_negatives)
            except:
                te_neg_idx = np.random.choice(list(diff_movies.difference(tr_neg_idx)), size = n_test_negatives, replace = True)

            for i in tr_neg_idx:
                train_neg_list.append((usr_idx, i))

            for j in te_neg_idx:
                test_neg_list.append((usr_idx, j))

        
        # save the datasets
        try:
            # save the list as csv at the respective paths
            np.savetxt(trainneg_path, train_neg_list, delimiter ="\t", fmt ='% s')
            print("Train negative data created!")
        except:
            print("Train negative data request couldn't be processed!")

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