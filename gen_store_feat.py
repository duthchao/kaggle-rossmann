import pandas as pd
import numpy as np
from datetime import datetime
from time import time


def convert_comp_date(row):
    if pd.isnull(row['CompetitionOpenSinceYear']):
        return datetime(year=2020,month=1,day=1)
    else:
        return datetime(year=int(row['CompetitionOpenSinceYear']),



def convert_promo2_date(row):
    if pd.isnull(row['Promo2SinceWeek']):
        return np.nan
    else:
        Y = '%4d'%row['Promo2SinceYear']
        Ww = '%02d1'%(row['Promo2SinceWeek'])
        return datetime.strptime(Y+Ww,'%Y%W%w')

def gen_store_feat(xgb=False):
	df_store = pd.read_csv('./data/store.csv',encoding='utf8')

	t = time()
	if xgb:
		print('gen xgb store feature...')

		raw_feats = df_store[['Store','CompetitionDistance','Promo2']]
		dumm_feats = pd.get_dummies(df_store[['StoreType','Assortment']])
		store_feat = pd.concat([raw_feats,dumm_feats],axis=1)
	else:
		print('gen fm store feature...')

		store_feat = df_store[['Store','CompetitionDistance','Promo2','StoreType','Assortment']]

	store_feat.to_hdf('store_feat.h5','table')
	print('total %.2f sec'%(time()-t))
	return 

	if __name__ == '__main__':
		gen_store_feat()



