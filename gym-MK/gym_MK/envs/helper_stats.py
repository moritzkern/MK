import pandas as pd
import numpy as np

df = pd.read_csv("/Users/andreferdinand/Desktop/MOPT/MK/gym-MK/gym_MK/envs/Products.csv")

class state():
    pass

for l,m in [["STORAGE_"+i, int(df["STORAGE"][df["PRODUCT"]==i])] for i in df["PRODUCT"]] + [["DEMAND_"+j, 0] for j in df["PRODUCT"][df["ENDPRODUCT"]]]:
    setattr(state,l,m)


class min_prod():
    pass

for l,m in [["MIN_"+j, int(df["MIN"][df["PRODUCT"]==j])] for j in df["PRODUCT"]]:
    setattr(min_prod,l,m)

class demand_dist():
    pass
for l,m in [["DDIST_"+j, df["DEMAND_DIST"][df["PRODUCT"]==j].to_string(index=False)] for j in df["PRODUCT"][df["ENDPRODUCT"]]]:
    setattr(demand_dist,l,m)

class demand_dist_parameter():
    pass
for l,m in [["DDISTPARA_"+j, eval(df["DIST_PARA"][df["PRODUCT"]==j].to_list()[0])] for j in df["PRODUCT"][df["ENDPRODUCT"]]]:
    setattr(demand_dist_parameter,l,m)



# >> def hello(name='World'): print('Hello %s' % name)

# >>> hello()
# Hello World

# >>> hello('Vist')
# Hello Vist

# >>> d = {'fun': hello}

# >>> d['fun']()
# Hello World

# >>> d['fun']('Vist')
# Hello Vist 