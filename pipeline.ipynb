#carga
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(dataframe_name, validation=False):
    df = pd.read_csv(dataframe_name+".csv")
    X = df.drop('test_result',axis=1) 
    y = df['test_result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
    
    dataset_train = pd.DataFrame({ 'feature0': X_train , 'label': y_train  })
    dataset_train.to_csv('train.csv',index=False)
    print("Dataset train.csv listo ...")

    dataset_test = pd.DataFrame({ 'feature0': X_test , 'label': y_test })
    dataset_test.to_csv('test.csv',index=False)
    print("Dataset test.csv listo ...")
    
    if validation == False:
        pass
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=101)

        dataset_train = pd.DataFrame({ 'feature0': X_train , 'label': y_train  })
        dataset_train.to_csv('sub_train.csv',index=False)
        print("Sub Dataset train.csv listo ...")

        dataset_test = pd.DataFrame({ 'feature0': X_test , 'label': y_test })
        dataset_test.to_csv('sub_test.csv',index=False)
        print("Sub Dataset test.csv listo ...")

#entrenar

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix

def training(train_name, test_name, filename='mlparams'):
    log_model = LogisticRegression()
    X_train = pd.read_csv(train_name+'.csv')
    X_test = pd.read_csv(test_name+'.csv')
    log_model.fit(X_train.feature0, X_train.label)
    y_pred = log_model.predict(X_test.feature0)
    
    pickle.dump(log_model, open(filename, 'wb'))
    print('Parametros escritos en archivo mlparams')

def accuracy(test, pred):
    return accuracy_score(test, pred)

def conf_matrix(test, pred):
    return confusion_matrix(test, pred)

#validar

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
import pickle

def validation(filename='mlparams', dataframe_name):
    log_test = pickle.load(open(filename, 'rb'))
    X_test = pd.read_csv(dataframe_name+'.csv')
    log_test.fit(X_test.feature0, X_test.label)
    y_pred_test = log_test.predict(X_test.feature0)
    
    return y_pred_test

#inferir
from flask import Flask
from flask import request
from sklearn.linear_model import LogisticRegression
import pickle
import logging
import sys
 
print(__name__)
app = Flask(__name__)

filename = "mlparams"
api_mlparams = pickle.load(open(filename, 'rb'))

logging.info(api_mlparams)

@app.route('/infer')
def infer():  
    reqX_1 = request.args.get('x_1')
    reqX_2 = request.args.get('x_2')
    x_1 = float(reqX_1)
    x_2 = float(reqX_2)
    print((x_1,type(x_1)),file=sys.stderr)
    print((x_2,type(x_2)),file=sys.stderr)
    return {'y':log_infer.predict([x_1, x_2]).item()}

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)
