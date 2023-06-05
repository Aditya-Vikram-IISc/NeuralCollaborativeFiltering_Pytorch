import numpy as np
import torch

# For evavluation, a batch of 100 items corresponding to each user to be passed. First item is the positive one, rest are negative
# pos_item is the positive item, and top_k_items are k items in decending order of their score


def hit(pos_item, top_k_items):
    if pos_item in top_k_items:
        return 1
    return 0


def ndcg(pos_item, top_k_items):
    if pos_item in top_k_items:
        index = top_k_items.index(pos_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def evaluate_metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)
        
        # get predicition of one batch of test_loader. (1 postive and rest all negatives)
        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().list()

        pos_item = item[0].item()
        HR.append(hit(pos_item, recommends))
        NDCG.append(ndcg(pos_item, recommends))
        

    return np.mean(HR), np.mean(NDCG)


