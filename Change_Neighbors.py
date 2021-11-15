from surprise import KNNBasic, SVD, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np

def run_knn(user_base, max_k):
    # Load the movielens-100k dataset (download it if needed),
    data_frame = pd.read_csv (r'ratings_small.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
    #df = pd.array(data, dtype=int)
    # print(df)
    reader_one = Reader(sep=',', rating_scale=(1.0, 5.0))
    raw_data = Dataset.load_from_df(data_frame[["userId", "movieId", "rating"]], reader_one)

    # Set the user_based parameter to true so we get our user based collaborative filtering
    sim_options_user_msd = {'name': 'msd', 'user_based': user_base}
    algo_msd = KNNBasic(k=max_k, sim_options=sim_options_user_msd)

    # Run 5-fold cross-validation and print results
    if user_base:
        print("User_Based\nK: " + str(max_k))
    else:
        print("Item_Based\nK: " + str(max_k))
    cross_validate(algo_msd, raw_data, measures=['RMSE'], cv=5, verbose=True)


run_knn(True, 20)
