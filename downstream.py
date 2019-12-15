from data.gen_data import build_X, build_Y, build_XY
from data.gen_data_config import gen_data_config


tr_s, tr_e = '2010/1/1', '2018/12/31'
te_s, te_e = '2019/1/1', '2019/12/31'
feature_days = 3

if __name__ == "__main__":
    
    tr_x, tr_y, tr_result = build_XY(tr_s, tr_e, 
                                     gen_data_config['computed_features'], 
                                     feature_days)
    te_x, te_y, te_result = build_XY(te_s, te_e, 
                                     gen_data_config['computed_features'], 
                                     feature_days)
    