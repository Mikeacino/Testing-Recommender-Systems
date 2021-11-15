from surprise import KNNBasic, SVD, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np


# Load the movielens-100k dataset (download it if needed),
data_frame = pd.read_csv (r'ratings_small.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
#df = pd.array(data, dtype=int)
# print(df)
reader_one = Reader(sep=',', rating_scale=(1.0, 5.0))
raw_data = Dataset.load_from_df(data_frame[["userId", "movieId", "rating"]], reader_one)

# Set the user_based parameter to true so we get our user based collaborative filtering
sim_options_user_cosine = {'name': 'cosine', 'user_based': True}
sim_options_user_msd = {'name': 'msd', 'user_based': True}
sim_options_user_pearson = {'name': 'pearson_baseline', 'user_based': True}

algo_user_cosine = KNNBasic(sim_options=sim_options_user_cosine)
algo_user_msd = KNNBasic(sim_options=sim_options_user_msd)
algo_user_pearson = KNNBasic(sim_options=sim_options_user_pearson)

# Run 5-fold cross-validation and print results
print("User_Based and Cosine")
cross_validate(algo_user_cosine, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("User_Based and MSD")
cross_validate(algo_user_msd, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("User_Based and Pearson_Baseline")
cross_validate(algo_user_pearson, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Set the user_based parameter to false so we get our user based collaborative filtering
sim_options_item_cosine = {'name': 'cosine', 'user_based': False}
sim_options_item_msd = {'name': 'msd', 'user_based': False}
sim_options_item_pearson = {'name': 'pearson_baseline', 'user_based': False}

algo_item_cosine = KNNBasic(sim_options=sim_options_item_cosine)
algo_item_msd = KNNBasic(sim_options=sim_options_item_msd)
algo_item_pearson = KNNBasic(sim_options=sim_options_item_pearson)

print("Item_Based and Cosine")
cross_validate(algo_item_cosine, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Item_Based and MSD")
cross_validate(algo_item_msd, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Item_Based and Pearson_Baseline")
cross_validate(algo_item_pearson, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
