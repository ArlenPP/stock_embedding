from sys import modules, path, stderr, stdout
path.append("./data")
path.append("./")
import os
from bisect import bisect_left, bisect_right
import numpy as np
from pickle import dump, load
from datetime import date, timedelta, datetime
from database import StockDB
from db_config import db_config
from gen_data_config import gen_data_config
import feature

fitx = []
result = []
base_path = gen_data_config['base_path']

def generate_feature(fitx_df, computed_features): # {{{
    data = fitx_df.to_dict('records')
    numerical = [v for v in data[0].keys() if v not in ['Date']]

    for v in data:
        v['Date'] = v['Date'].strftime("%Y/%m/%d")
        for f in numerical:
            if(v[f] == ''):
                continue
            v[f] = float(v[f])
    for f in computed_features:
        getattr(modules['feature'], f)(data, f)
    
    return data
    # }}}

def date_to_index(dates, date, greater=True): # {{{
    if (greater):
        return bisect_left(dates, date)
    else:
        return bisect_right(dates, date) - 1
    # }}}

def build_X(start_date, end_date, selected_features, feature_days): # {{{
    start_date = datetime.strptime(start_date, '%Y/%m/%d').strftime('%Y/%m/%d')
    end_date = datetime.strptime(end_date, '%Y/%m/%d').strftime('%Y/%m/%d')
    
    X = []
    dates = [v['Date'] for v in fitx]
    start_index = date_to_index(dates, start_date)
    end_index   = date_to_index(dates,   end_date, False)

    for i in range(start_index, end_index+1):
        X.append([])
        if 0 == feature_days:
            X[-1].extend([fitx[i][v] for v in selected_features])
        else:
            for j in range(i - feature_days, i):
                X[-1].extend([fitx[j][v] for v in selected_features])
    return np.array(X)
    # }}}

def build_Y(start_date, end_date):
    subresult = result.loc[start_date : end_date]
    Y = []
    for i in range(len(subresult)):
        y = 0
        if(subresult['Call_take_profit'][i] == 1):
            y = 1
        elif(subresult['Put_take_profit'][i] == 1):
            y = -1
        Y.append(y)
    return np.array(Y), subresult

def build_XY(start_date, end_date, selected_features, feature_days):
    X = build_X(start_date, end_date, selected_features, feature_days)
    Y, result = build_Y(start_date, end_date)
    return X, Y, result

def update_local_data(today):
    fitx_pkl = {}
    db = StockDB(**db_config)
    fitx_pkl['fitx_df'] = db.read_data("1999-01-01", today, True)
    fitx_pkl['result'] = db.read_result("1999-01-01", today, gen_data_config['result_id'])
    dump(fitx_pkl, open(base_path+today+".pkl", "wb"))
    return fitx_pkl
def check_file_date(today):
    file_list = os.listdir(base_path)
    flag = True
    if(0 == len(file_list)): 
        return False
    for file in file_list:
        if(today != os.path.splitext(file)[0]):
            flag = False
            os.remove(base_path+file)
    return flag

def smart_update():
    global fitx, result
    today = date.today().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M")
    if not (check_file_date(today) and now < "19:05"):
        fitx_pkl = update_local_data(today)
    else:
        with open(base_path+today+".pkl", 'rb') as f: 
            fitx_pkl = load(f)
    fitx = generate_feature(fitx_pkl['fitx_df'], gen_data_config['computed_features'])
    result = fitx_pkl['result']

smart_update()