import pandas as pd

df = pd.read_csv("/Users/andreferdinand/Desktop/MOPT/MK/gym-MK/gym_MK/envs/Products.csv")

class State():
    pass
state = State()

for l,m in [["STORAGE_"+i, int(df["STORAGE"][df["PRODUCT"]==i])] for i in df["PRODUCT"]] + [["DEMAND_"+j, 0] for j in df["PRODUCT"][df["ENDPRODUCT"]]]:
    setattr(State,l,m)