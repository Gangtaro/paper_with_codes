import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class BaseDataset(Dataset):
    def __init__(self, data_path:str) -> None:
        super().__init__()
        # load data
        ## u.data: The full u data set, 100000 ratings by 943 users on 1682 items; The time stamps are unix seconds since 1/1/1970 UTC
        ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings = pd.read_csv( os.path.join(data_path, 'u.data'), sep = '\t', names=ratings_cols, encoding='latin-1')

        ## u.info: The number of users, items, and ratings in the u data set
        self.info = pd.read_csv(os.path.join(data_path,'u.info'), sep='\s+', names=['number', 'type'], encoding='latin-1')

        ## u.genre: A list of the genres
        self.genres = pd.read_csv(os.path.join(data_path,'u.genre'), sep = '|', names = ['genre', 'number'], encoding='latin-1')

        ## u.item: Information about the items (movies)
        movies_cols = [
			'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
		] + self.genres['genre'].tolist()
        self.items = pd.read_csv(os.path.join(data_path,'u.item'), sep='|', names=movies_cols, encoding='latin-1')

		## u.user: Demographic information about the users
        users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.users = pd.read_csv(os.path.join(data_path, 'u.user'), sep='|', names=users_cols, encoding='latin-1')

        ## u.occupation: A list of the occupations.
        self.occupation = pd.read_csv(os.path.join(data_path,'u.occupation'), sep='\s+', names=['occupation'], encoding='latin-1')

        # preprocessing
        ## ~

    def __getitems__(self, idx):
        return 

    def __len__(self):
        return 
