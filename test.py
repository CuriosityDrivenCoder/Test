## En son sadelestirilien ve calisan kod

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cx_Oracle

def get_data_from_db():
    """Database connection and data retrieval"""
    tns = "url"
    user = "usr" #KULLANICI ADI
    passw = "pass" #ŞİFRE 
    con = cx_Oracle.connect(user, passw, tns)
    cur = con.cursor()
    
    query = """
    SELECT 
    BRANCH_ID as PARTY_ID, DATA_DATE as DATA_DATE, WORKING_CAPACITY_AMOUNT_TL as WORKING_CAPACITY_AMOUNT_TL, NET_PROFIT_AMOUNT_TL as NET_PROFIT_AMOUNT_TL, COST_INCOME_RATIO as COST_INCOME_RATIO, NOF_ACTIVE_CUSTOMER as NOF_ACTIVE_CUSTOMER, NPL_CREDIT_CASH_LOAN_RATIO as NPL_CREDIT_CASH_LOAN_RATIO, OTHER_WORK_CAP_AMOUNT_TL as OTHER_WORK_CAP_AMOUNT_TL, NOF_FINANCIAL_TRANSACTION_L12M as NOF_FINANCIAL_TRANSACTION_L12M, CAPACITY_USAGE_RATIO as CAPACITY_USAGE_RATIO, DIGITAL_CUSTOMER_RATIO as DIGITAL_CUSTOMER_RATIO, NEW_ACTIVE_CUSTOMER_RATIO_L12M as NEW_ACTIVE_CUSTOMER_RATIO_L12M, ACTIVE_PRODUCT_CUSTOMER_RATIO as ACTIVE_PRODUCT_CUSTOMER_RATIO, CREDIT_MARKET_SHARE_EXCL_BRCH as CREDIT_MARKET_SHARE_EXCL_BRCH, DEPOSIT_MARKET_SHARE_EXCL_BRCH as DEPOSIT_MARKET_SHARE_EXCL_BRCH, DISTRICT_ADULT_BRANCH_RATIO as DISTRICT_ADULT_BRANCH_RATIO, NOF_COMPETITOR_BRANCH_NEARBY as NOF_COMPETITOR_BRANCH_NEARBY, WORK_CAP_AMOUNT_TL_RATIO_L12M as WORK_CAP_AMOUNT_TL_RATIO_L12M, ACTIVE_CUSTOMER_RATIO_L12M as ACTIVE_CUSTOMER_RATIO_L12M, BUDGET_WORK_CAP_AMT_TL_RATIO as BUDGET_WORK_CAP_AMT_TL_RATIO, CUSTOMER_PRODUCTIVITY_RATIO as CUSTOMER_PRODUCTIVITY_RATIO, NOF_BRANCH_IN_MICRO_MARKET as NOF_BRANCH_IN_MICRO_MARKET, MICRO_MARKET_POTENTIAL_RATIO as MICRO_MARKET_POTENTIAL_RATIO, COMPETITOR_STATE_ID as COMPETITOR_STATE_ID
    FROM SCHEMA.TABLENAME
    """
    
    cur.execute(query)
    data_m = cur.fetchall()
    
    df = pd.DataFrame(data_m)
    df.columns = [i[0] for i in cur.description]
    
    return df

def prepare_data(df):
    """Data preparation and scaling"""
    data_date = df["DATA_DATE"]
    df = df.drop(columns=['DATA_DATE'])
    branch_data = df.rename(columns={'PARTY_ID': 'KOD'}).copy()
    var_3 = list(branch_data.columns)[1:]
    branch_data.fillna(0, inplace=True)
    
    scaler = StandardScaler()
    scale_val = scaler.fit_transform(branch_data[var_3])
    
    return branch_data, var_3, scale_val

def find_optimal_clusters(scale_val, branch_data, info_count=1):
    """Find optimal number of clusters for each variable"""
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 1000, "random_state": 42}
    num_of_cluster = {}
    slice_count = 20
    
    for i in range(0, scale_val.shape[1]):
        sse = []   
        for k in range(1, slice_count):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(scale_val[:, i].reshape(len(scale_val), 1))
            sse.append(kmeans.inertia_)
            
        difference = abs(np.diff(np.array(sse)))
        num_of_cluster[branch_data.columns.values[i + info_count]] = np.sum(difference > difference.sum()*0.01)
    
    num_of_cluster = {k: min(max(v, 5), 7) for k, v in num_of_cluster.items()}
    num_of_cluster["COMPETITOR_STATE_ID"] = 5  # Override for this specific column
    
    return num_of_cluster

def cluster_and_score(branch_data, var_list, scale_val, num_of_cluster, ratios, info_count=1):
    """Apply K-means clustering and calculate scores"""
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 1000, "random_state": 42}
    
    for i, area in enumerate(var_list):
        kmeans = KMeans(n_clusters=num_of_cluster.get(area), **kmeans_kwargs)
        kmeans.fit(scale_val[:, i].reshape(-1, 1))
        
        # Sort data and calculate scores
        df = pd.DataFrame({
            'KOD': branch_data['KOD'],
            area: branch_data[area],
            'DILIM': kmeans.labels_,
            'OLCK': scale_val[:, i]
        }).sort_values(by=area, ascending=ratios[i]).reset_index(drop=True)
        
        # Calculate scores
        skor = 0
        df[area + '_SKOR'] = 0
        for j in range(len(df) - 1):
            df.at[j, area + '_SKOR'] = skor
            if df.at[j, 'DILIM'] != df.at[j + 1, 'DILIM']:
                skor += 1
        df.at[len(df) - 1, area + '_SKOR'] = skor
        
        # Merge scores back to branch_data
        branch_data = pd.merge(branch_data, df[['KOD', area + '_SKOR']], on='KOD', how='inner')
    
    return branch_data

def manipulate_clusters(branch_data, var_3, num_of_cluster, ratios, alt_limit=0.1, ust_limit=0.25):
    """Manipulate clusters based on limits"""
    recluster = pd.DataFrame({
        "VARIABLE": var_3,
        "RATIOS": ratios,
        "IS_MANUPULATED": np.zeros(len(ratios)),
        "CLUSTER_NO": np.zeros(len(ratios)),
        "SEMI_CLUSTER": np.zeros(len(ratios))
    })
    
    for x in var_3:
        mem_rat=0
        for i in range(num_of_cluster[x]):
            if i==0:
                if branch_data[branch_data[x+'_SKOR']==i].KOD.count()/branch_data.KOD.count()>=alt_limit:
                    break
                elif branch_data[branch_data[x+'_SKOR']==i].KOD.count()/branch_data.KOD.count()<alt_limit:
                    mem_rat=mem_rat+branch_data[branch_data[x+'_SKOR']==i].KOD.count()/branch_data.KOD.count()
    
            else:
                if mem_rat<alt_limit:
                    mem_rat=mem_rat+branch_data[branch_data[x+'_SKOR']==i].KOD.count()/branch_data.KOD.count()
                    if mem_rat>=ust_limit:
                        recluster.loc[recluster.VARIABLE==x,"IS_MANUPULATED"]=1
                        if i==num_of_cluster[x]-1:
                            recluster.loc[recluster.VARIABLE==x,"CLUSTER_NO"]=i-1
                        elif i<=num_of_cluster[x]-2 and x=='OTHER_WORK_CAP_AMOUNT_TL':
                            recluster.loc[recluster.VARIABLE==x,"CLUSTER_NO"]=i+1
                        else:
                            recluster.loc[recluster.VARIABLE==x,"CLUSTER_NO"]=i
    
                        if x=='OTHER_WORK_CAP_AMOUNT_TL':
                            recluster.loc[recluster.VARIABLE==x,"SEMI_CLUSTER"]=0
                        else:
                            recluster.loc[recluster.VARIABLE==x,"SEMI_CLUSTER"]=1
                        break
                    elif mem_rat>=alt_limit:
                        recluster.loc[recluster.VARIABLE==x,"IS_MANUPULATED"]=1
                        if i==num_of_cluster[x]-1:
                            recluster.loc[recluster.VARIABLE==x,"CLUSTER_NO"]=i-1
                        elif i<=num_of_cluster[x]-2 and x=='OTHER_WORK_CAP_AMOUNT_TL':
                            recluster.loc[recluster.VARIABLE==x,"CLUSTER_NO"]=i+1
                        else:
                            recluster.loc[recluster.VARIABLE==x,"CLUSTER_NO"]=i
                        break
    recluster.loc[recluster.VARIABLE=="NOF_RANCH_IN_MICRO_MARKET","IS_MANUPULATED"]=0 
    recluster.loc[recluster.VARIABLE=="COMPETITOR_STATE_ID","IS_MANUPULATED"]=0      
    
    return recluster

def apply_manipulation(branch_data, recluster, var_3):
    """Apply the manipulation to the data"""
    data2 = branch_data.copy()
    for x in var_3:
        y = x + "_SKOR"
        recluster_x = recluster[recluster.VARIABLE == x]
        
        if recluster_x.IS_MANUPULATED.values == 1:
            cluster_no = int(recluster_x.CLUSTER_NO)
            is_ratio = recluster_x.RATIOS.values
            semi_cluster = recluster_x.SEMI_CLUSTER.values == 1
            
            if semi_cluster:
                mean_value = data2[data2[y] == cluster_no][x].mean()
                data2[x] = data2[x].apply(lambda j: data2[x].max() * 100 if j >= mean_value else (data2[x].min() / 100 if is_ratio else j))
            else:
                if is_ratio:
                    max_value = data2[data2[y] <= cluster_no][x].max()
                    data2[x] = data2[x].apply(lambda j: data2[x].min() / 100 if j <= max_value else j)
                else:
                    min_value = data2[data2[y] <= cluster_no][x].min()
                    data2[x] = data2[x].apply(lambda j: data2[x].max() * 100 if j >= min_value else j)
    
    return data2

def final_scoring(branch_data, var_3, var_skor, ratios, coefficients):
    """Calculate final model score"""
    mmscaler = MinMaxScaler()
    olck_degler = []
    for x in var_skor:
        branch_data = pd.concat([branch_data, pd.DataFrame(mmscaler.fit_transform(branch_data[[x]]), columns=[x + '_OLCK'])], axis=1)
        olck_degler.append(x + '_OLCK')
    
    branch_data['MODEL_SCORE'] = (branch_data[olck_degler] * coefficients).sum(axis=1)
    
    return branch_data, olck_degler

def main():
    df = get_data_from_db()
    branch_data, var_3, scale_val = prepare_data(df)
    
    ratios = [
        False, False, True, False, True, False, False, False, True, False,
        False, False, False, False, False, False, False, False, False, True,
        False, True
    ]
    
    num_of_cluster = find_optimal_clusters(scale_val, branch_data)
    branch_data = cluster_and_score(branch_data, var_3, scale_val, num_of_cluster, ratios)
    
    var_skor = [var + '_SKOR' for var in var_3]
    recluster = manipulate_clusters(branch_data, var_3, num_of_cluster, ratios)
    data2 = apply_manipulation(branch_data, recluster, var_3)
    
    branch_data_2 = data2.copy()
    for x in var_skor:
        branch_data_2 = branch_data_2.drop(x, axis=1)
        branch_data = branch_data.drop(x, axis=1)
    
    scale_val_2 = StandardScaler().fit_transform(branch_data_2[var_3])
    branch_data = cluster_and_score(branch_data, var_3, scale_val_2, num_of_cluster, ratios)
    
    coefficients = [12, 12, 10, 6, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 4, 3, 4, 4, 3]
    branch_data, olck_degler = final_scoring(branch_data, var_3, var_skor, ratios, coefficients)
    
    branch_data_2 = branch_data.copy()
    skor_column = [r + '_SCORE' for r in var_3]
    
    for i in range(len(coefficients)):
        branch_data_2[skor_column[i]] = branch_data_2[olck_degler[i]] * coefficients[i]
    
    branch_data_2 = branch_data_2.drop(var_skor + olck_degler, axis=1)
    
    column_mapping = {
        'KOD': 'BRANCH_ID',
        'WORKING_CAPACITY_AMOUNT_TL_SCORE': 'WORK_CAP_AMOUNT_TL_SCORE',
        'NPL_CREDIT_CASH_LOAN_RATIO_SCORE': 'NPL_CR_CSH_LOAN_RATIO_SCORE',
        'NOF_FINANCIAL_TRANSACTION_L12M_SCORE': 'NOF_FINANCIAL_TRX_SCORE_L12M',
        'NEW_ACTIVE_CUSTOMER_RATIO_L12M_SCORE': 'NEW_ACTV_CUST_RATIO_SCORE_L12M',
        'ACTIVE_PRODUCT_CUSTOMER_RATIO_SCORE': 'ACTIVE_PRD_CUST_RATIO_SCORE',
        'CREDIT_MARKET_SHARE_EXCL_BRCH_SCORE': 'CR_MKT_SHARE_EXCL_BRCH_SCORE',
        'DEPOSIT_MARKET_SHARE_EXCL_BRCH_SCORE': 'DEPO_MKT_SHARE_EXCL_BRCH_SCORE',
        'DISTRICT_ADULT_BRANCH_RATIO_SCORE': 'DISTRICT_ADLT_BRCH_RATIO_SCORE',
        'NOF_COMPETITOR_BRANCH_NEARBY_SCORE': 'NOF_COMPT_BRCH_NEARBY_SCORE',
        'WORK_CAP_AMOUNT_TL_RATIO_L12M_SCORE': 'WORK_CAP_AMT_RATIO_SCORE_L12M',
        'ACTIVE_CUSTOMER_RATIO_L12M_SCORE': 'ACTIVE_CUST_RATIO_SCORE_L12M',
        'BUDGET_WORK_CAP_AMT_TL_RATIO_SCORE': 'BDG_WORK_CAP_AMT_RATIO_SCORE',
        'CUSTOMER_PRODUCTIVITY_RATIO_SCORE': 'CUSTOMER_PRDTY_RATIO_SCORE',
        'NOF_BRANCH_IN_MICRO_MARKET_SCORE': 'NOF_BRANCH_IN_MICRO_MKT_SCORE',
        'MICRO_MARKET_POTENTIAL_RATIO_SCORE': 'MICRO_MARKET_POT_RATIO_SCORE'
    }
    branch_data_2 = branch_data_2.rename(columns=column_mapping)
    
    result_df = branch_data_2.drop(columns=['NET_PROFIT_AMOUNT_TL','COST_INCOME_RATIO','NOF_ACTIVE_CUSTOMER','OTHER_WORK_CAP_AMOUNT_TL','CAPACITY_USAGE_RATIO','DIGITAL_CUSTOMER_RATIO','COMPETITOR_STATE_ID','WORKING_CAPACITY_AMOUNT_TL','NPL_CREDIT_CASH_LOAN_RATIO','NOF_FINANCIAL_TRANSACTION_L12M','NEW_ACTIVE_CUSTOMER_RATIO_L12M','ACTIVE_PRODUCT_CUSTOMER_RATIO','CREDIT_MARKET_SHARE_EXCL_BRCH','DEPOSIT_MARKET_SHARE_EXCL_BRCH','DISTRICT_ADULT_BRANCH_RATIO','NOF_COMPETITOR_BRANCH_NEARBY','WORK_CAP_AMOUNT_TL_RATIO_L12M','ACTIVE_CUSTOMER_RATIO_L12M','BUDGET_WORK_CAP_AMT_TL_RATIO','CUSTOMER_PRODUCTIVITY_RATIO','NOF_BRANCH_IN_MICRO_MARKET','MICRO_MARKET_POTENTIAL_RATIO'])
    
    return result_df

if __name__ == "__main__":
    result_df = main()
