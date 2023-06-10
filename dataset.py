from torch.utils.data import Dataset
import random

class NCFTraining_Dataset(Dataset):
    def __init__(self, ratings_train_pos_path:str, ratings_train_neg_path:str):
        super(NCFTraining_Dataset, self).__init__()
        positive_data = self.dataprocessor(ratings_train_pos_path, target = 1.0)
        negative_data = self.dataprocessor(ratings_train_neg_path, target = 0.0)
        self.traindataset = positive_data + negative_data
        random.shuffle(self.traindataset)


    def __len__(self):
        return len(self.traindataset)

    def __getitem__(self, index):
        # return userid, movieid, interaction_label
        return self.traindataset[index][0], self.traindataset[index][1], self.traindataset[index][2]


    def dataprocessor(self, path, target = 1.0):
        data_list = []
        with open(path, "r") as f:
            textline = f.readline()
            while textline != None and textline != "":
                arr = textline.split("\t")
                usr_idx, item_idx = int(arr[0]), int(arr[1])
                data_list.append((usr_idx, item_idx, target))
                textline = f.readline()
        return data_list


class NCFTesting_Dataset(Dataset):
    def __init__(self, ratings_test_path:str):
        super(NCFTesting_Dataset, self).__init__()
        self.testdataset = self.datapreprocessor(path = ratings_test_path)


    def __len__(self):
        return len(self.testdataset)
    
    def __getitem__(self, index):
    # return userid, movieid
        return  self.testdataset[index][0], self.testdataset[index][1]


    def datapreprocessor(self, path):
        data_list = []
        with open(path, "r") as f:
            textline = f.readline()
            while textline != None and textline != "":
                arr = textline.split("\t")
                user_id, mov_id = tuple(map(int, arr[0][1:-1].split(",")))
                data_list.append((user_id, mov_id))
                for x in arr[1:]:
                    data_list.append((user_id, int(x)))
                textline =  f.readline()
                if textline == "\n":
                    textline =  f.readline()
        return data_list
        
    
if __name__ == "__main__":
    ncf_traindataset = NCFTraining_Dataset("data/ratings_train_pos", "data/ratings_train_neg")
    ncf_testdataset = NCFTesting_Dataset("data/ratings_test_neg")
    print("Size of train dataset:", len(ncf_traindataset))
    print("Size of test dataset:", len(ncf_testdataset))