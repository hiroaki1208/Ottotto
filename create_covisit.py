print("hello")
a = 1
b = 2
print(a+b)

import pandas as pd


VER = 8

import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
from copy import deepcopy
# import itertools
import cudf, itertools
print('We will use RAPIDS version',cudf.__version__)

def create_covisit_type():
    '''
    co-visitation(type weighted)作成
    - create pairsの前に重複削除
        - 重複削除のときにweight sumを取るのはどう？
    - create pairsの前にaid_xを限定したto-mergeを作成することでメモリ削減
    '''

    # USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
    # THRED_H = 1
    DISK_PIECES = 4
    SIZE = 1.86e6/DISK_PIECES
    # TOP = 20

    # COMPUTE IN PARTS FOR MEMORY MANGEMENT
    for PART in range(DISK_PIECES):
#         if PART > 0: break

        print()
        print('### DISK PART',PART+1)

        # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
        # => OUTER CHUNKS
        for j in range(6):
#             if j > 0: break

            a = j*CHUNK
            b = min( (j+1)*CHUNK, len(files) )
            print(f'Processing files {a} thru {b-1} in groups of {READ_CT}...')

            # => INNER CHUNKS
            for k in range(a,b,READ_CT):
#                 if k > a+READ_CT: break

                # READ FILE
    #             print(files[k])
                df = [read_file(files[k])]
                for i in range(1,READ_CT): 
    #                 print(files[k+i])
                    if k+i<b: df.append( read_file(files[k+i]) )
                df = cudf.concat(df,ignore_index=True,axis=0)

                # ==================
                # ver0 追記
                # sessionをﾗｸﾞごとに再分割
                if is_split:
        #             df['ts_diff'] = df.groupby('session').ts.diff().fillna(0) # NG
                    df = df.sort_values(['session','ts']).reset_index(drop=True)
                    df['d'] = df.ts.diff().fillna(0)
                    df.loc[ df.session.diff()!=0, 'd'] = 0 # session感のdiff=0

                    df["d_f"] = (df.d > THRED_H*60*60).astype("int8").fillna(0)
                    df["session_new"] = df.groupby("session").d_f.cumsum()    
                    df["session_n"] = (df["session"]*1_000_000 + df["session_new"]).astype(int)
                    nu = df.session_n.nunique()
                    print(f"all session num: {nu:,}")

                    sz = df.groupby("session_n").size().reset_index().rename(columns= {0: "sz"})
                    sz_1 = sz.loc[sz.sz == 1, "session_n"].values    
                    sz_1_l = len(sz_1)
                    sz_1_r = sz_1_l / nu * 100
                    print(f"len session sz=1: {sz_1_l:,}")
                    print(f"len session sz=1: {sz_1_r: .2f}(%)")

                    cols = [
                        "session_n",
                        "aid",
                        "ts",
                        "type",
                    ]
                    df = df.loc[~df.session_n.isin(sz_1), cols]\
                        .rename(columns= {"session_n": "session"}).copy()

                # ver0 追記
                # ==================            
                # create pairsの前に重複削除
                df["wgt"] = df["type"].map(TYPE_WEIGHT)
                df.wgt = df.wgt.astype("float32")
                df = df.groupby(["session", "aid"]).wgt.sum()
                df = df.reset_index()
                
                # create pairs
                df_merge = df.loc[(df.aid >= PART*SIZE)&(df.aid < (PART+1)*SIZE)]
                df = df_merge.merge(df, on= "session")
                del df_merge
                _ = gc.collect()
                df = df.loc[  (df.aid_x != df.aid_y) ]
                df = df[["aid_x", "aid_y", "wgt_y"]]
                df = df.groupby(["aid_x", "aid_y"]).wgt_y.sum()
#                 df = df.set_index(["aid_x", "aid_y"]).wgt_y # NG
                
                # COMBINE INNER CHUNKS
                if k==a: tmp2 = df
                else: tmp2 = tmp2.add(df, fill_value=0)
                print(k,', ',end='')
                
            print()
            # COMBINE OUTER CHUNKS
            if a==0: tmp = tmp2
            else: tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, df
            gc.collect()
            
        # CONVERT MATRIX TO DICTIONARY
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(['aid_x','wgt_y'],ascending=[True,False])
        # SAVE TOP 40
        tmp = tmp.reset_index(drop=True)
        tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
        tmp = tmp.loc[tmp.n<TOP].drop('n',axis=1)
#         # SAVE PART TO DISK (convert to pandas first uses less memory)
        tmp.to_pandas().to_parquet(f'top_{TOP}_type_w{ns}_v{VER}_{PART}.pqt')

    return

# %%time
# CACHE FUNCTIONS
def read_file(f):
    return cudf.DataFrame( data_cache[f] )
def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts/1000).astype('int32')
    df['type'] = df['type'].map(type_labels).astype('int8')
    return df

# CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
data_cache = {}
type_labels = {'clicks':0, 'carts':1, 'orders':2}
# files = sorted(glob.glob('../input/otto-chunk-data-inparquet-format/train_parquet/*'))
files = sorted(glob.glob('../input/otto-validation/*_parquet/*'))
for f in files: data_cache[f] = read_file_to_cache(f)

# CHUNK PARAMETERS
READ_CT = 5
CHUNK = int( np.ceil( len(files)/6 ))
print(f'We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.')

# common parameters
THRED_H = 1 # N時間以上離れたとき別sessionとみなす
TOP = 20 # 上位Nのaidを返す
is_split = True
ns = "" if is_split else "_ns"
TYPE_WEIGHT = {0:1, 1:6, 2:3}

# %%time
# create_covisit_type()