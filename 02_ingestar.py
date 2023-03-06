import pandas as pd
from sklearn.model_selection import train_test_split

def data_structure(train_1, train_2, df_name):
    _df = train_1.copy()
    _df['test_result'] = train_2
    _df.to_csv(df_name+'.csv', encoding='utf-8')

def split_data(dataframe_name, validation=False, ss=0.1):
    datasets = pd.DataFrame()
    df = pd.read_csv(dataframe_name)
    X = df.drop('test_result',axis=1) 
    y = df['test_result']
    
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=ss, random_state=101)
    
    if validation == False:
        setname_01 = 'train'
        setname_02 = 'test'
    else:
        setname_01 = 'train_01'
        setname_02 = 'train_02'
        
    data_structure(X_train, y_train, setname_01)
    data_structure(X_test, y_test, setname_02)

split_data('hearing_test.csv', validation=False, ss=0.1)
split_data('train.csv', validation=True, ss=0.3)
