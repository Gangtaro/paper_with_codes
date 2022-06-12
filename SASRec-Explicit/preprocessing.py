import pandas as pd
import numpy as np

print("-- load raw data")
df_ratings = pd.read_csv("../data/ml-1m/ratings.dat", names= ["user", "item", "rating", "time"], sep = "::", engine="python")
print("-- preprocessing ...")
df_ratings.to_csv("../data/ml-1m/train_ratings.csv", index=False)
print("-- done !")