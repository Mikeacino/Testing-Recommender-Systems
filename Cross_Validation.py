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

# SVD with biased set to false is essentially PMF
algo_pmf = SVD(biased = False)

# Set the user_based parameter to true so we get our user based collaborative filtering
sim_options_user_based = {'user_based': True}
algo_user_based = KNNBasic(sim_options=sim_options_user_based)

# Set the user_based parameter to false for our item based collaborative filtering
sim_options_item_based = {'user_based': False}
algo_item_based = KNNBasic(sim_options=sim_options_item_based)

# Run 5-fold cross-validation and print results
cross_validate(algo_pmf, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cross_validate(algo_user_based, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cross_validate(algo_item_based, raw_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
