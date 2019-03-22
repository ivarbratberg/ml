import pandas as pd
import numpy as np
# Mean encode feature
def mean_encode(m,col,cols, aggregate_name):
    grouped = m.groupby(cols, as_index=False).agg({ col: 'mean'})
    print("Hei")
    grouped = grouped.rename(columns={col : aggregate_name})
    ret = pd.merge(m,grouped,on=cols, how='left').fillna(0)
    ret[aggregate_name] = ret[aggregate_name].astype(np.float16)
    return ret



# Effort to copy directly using keys
def create_lags2(m, lag_column_name ,key_cols, value_col_name, lag):    
    unique_cols = key_cols + [lag_column_name]
    print("Creating lag for " + value_col_name + ' ' + str(lag))    
    # uniqe values for the unique cols which includeds tha lage column
    m_reduced = m[ unique_cols + [value_col_name] ].drop_duplicates()    
#     m_reduced.to_csv("d:/temp/uniq.csv")
#     # merge in the value we want to create lag for
#     m_reduced = pd.merge(m_reduced, m[unique_cols+[value_col_name]], on = unique_cols, how='left' )
    for lag in range(lag[0],lag[1]+1):           
        print("Lag " + str(lag))
        m_reduced[lag_column_name] -= lag
        # m_reduced.to_csv("d:/temp/uniq.csv",index=False)
        m = pd.merge(m,m_reduced, on=unique_cols,how='left').fillna(0)
        m_reduced[lag_column_name] += lag
        m = m.rename(index=str, columns={value_col_name+"_x" : value_col_name,
                value_col_name+"_y" : value_col_name+str(lag)})     
    return m
