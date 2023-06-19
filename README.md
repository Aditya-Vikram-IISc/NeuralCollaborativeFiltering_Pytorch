# NeuralCollaborativeFiltering_Pytorch
This is my end-to-end pytorch implementation of [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). The repo implements the  Neural matrix factorization model without pretraining.

## 1. ABSTRACT

The key to a personalized recommender system is in modelling users’ preference on items based on their past interactions (e.g., ratings and clicks), known as collaborative filtering. Among the various collaborative filtering techniques, matrix factorization is the most popular one, which projects users and items into a shared latent space, using a vector of latent features to represent a user or an item. Thereafter a user’s interaction on an item is modelled as the inner product of their latent vectors.

The proposed neural matrix factorization model, which ensembles MF and MLP under the NCF framework; it unifies the strengths of linearity of MF and non-linearity of MLP for modelling the user–item latent structures


## 2. DATASET CREATION

The repo uses the ratings.dat file from [MovieLens 1Million Dataset](https://grouplens.org/datasets/movielens/1m/). To create the train and test dataset, leave-one-out formulation has been implemented where for each user, we held-out his/ her latest interaction as the test set and utilized the remaining data for training. 

As the paper has been formulated as a Binary CLassification Problem, we sampled four negative instances per positive instance during training. For evaluation, we randomly sampled 100 items that are not interacted by the user, ranking the levae-one-out test item among the 100 items. The performance of a ranked list is judged by Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG).

To create the training & testing dataset, place the ratings.dat in /data folder run the following command:

>> python dataset_creation.py --file_path data/ratings.dat --trainpos_path data/ratings_train_pos --testpos_path data/ratings_test_pos --trainneg_path data/ratings_train_neg --testneg_path data/ratings_test_neg --n_train_negatives 4 --n_test_negatives 100