from torch.utils.data import Dataset
import random

class NCFTraining_Dataset(Dataset):
    def __init__(self, ratings_train_pos_path:str, ratings_train_neg_path:str):
        positive_data = self.dataprocessor(ratings_train_pos_path, target = 1.0)
        negative_data = self.dataprocessor(ratings_train_neg_path, target = 0.0)
        self.fulldataset = positive_data + negative_data
        random.shuffle(self.fulldataset)


    def __len__(self):
        return len(self.fulldataset)

    def __getitem__(self, index):
        # return userid, movieid, interaction_label
        return self.fulldataset[index][0], self.fulldataset[index][1], self.fulldataset[index][2]


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
    
if __name__ == "__main__":
    ncf_dataset = NCFTraining_Dataset("data/ratings_train_pos", "data/ratings_train_neg")
    print("Size of dataset:", len(ncf_dataset))