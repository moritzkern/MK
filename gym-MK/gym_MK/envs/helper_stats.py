import pandas as pd

df = pd.read_csv("/Users/andreferdinand/Desktop/MOPT/MK/gym-MK/gym_MK/envs/Products.csv")

class state():
    pass

for l,m in [["STORAGE_"+i, int(df["STORAGE"][df["PRODUCT"]==i])] for i in df["PRODUCT"]] + [["DEMAND_"+j, 0] for j in df["PRODUCT"][df["ENDPRODUCT"]]]:
    setattr(state,l,m)


class min_prod():
    pass

for l,m in [["MIN_"+j, int(df["MIN"][df["PRODUCT"]==j])] for j in df["PRODUCT"]]:
    setattr(min_prod,l,m)

