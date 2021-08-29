import numpy as np 
import pandas as pd

d1 = {"PRODUCT_ID": 0,"PRODUCT": "DOUGH", "STORAGE": 30 , "ENDPRODUCT": False,"MIN": 100, "PROCESS_00": "BREAD", "PROCESS_01": "BUNS"}
d2 = {"PRODUCT_ID": 1,"PRODUCT": "BREAD", "STORAGE": 10, "ENDPRODUCT": True,"DEMAND_DIST": "NORMAL" , "DIST_PARA": (1,0.1), "MIN": 70}
d3 = {"PRODUCT_ID": 2,"PRODUCT": "BUNS", "STORAGE": 10, "ENDPRODUCT": False, "MIN": 70, "PROCESS_00": "BUNS1", "PROCESS_01": "BUNS2", "PROCESS_02": "BUNS3"}
d4 = {"PRODUCT_ID": 3,"PRODUCT": "BUNS1", "STORAGE": 10, "ENDPRODUCT": True, "DEMAND_DIST": "NORMAL","DIST_PARA": (1,0.1) ,"MIN": 10}
d5 = {"PRODUCT_ID": 4,"PRODUCT": "BUNS2", "STORAGE": 10, "ENDPRODUCT": True, "DEMAND_DIST": "NORMAL","DIST_PARA": (1,0.1),"MIN": 10}
d6 = {"PRODUCT_ID": 5,"PRODUCT": "BUNS3", "STORAGE": 10, "ENDPRODUCT": True, "DEMAND_DIST": "NORMAL", "DIST_PARA": (1,0.1), "MIN": 10}
df= pd.DataFrame([d1,d2,d3,d4,d5,d6])
df.set_index("PRODUCT_ID")

csv = df.to_csv("/Users/andreferdinand/Desktop/MOPT/MK/gym-MK/gym_MK/envs/Products.csv",index=False)

