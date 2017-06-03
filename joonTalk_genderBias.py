import types
import numpy as np
import pickle

def row2XY(row, features, num_features, value_field):
    x = np.ndarray((1, num_features))
    fcount = 0
    for feature in features:
        if ( isinstance(row[feature], types.ListType) ):
            for f in row[feature]:
                x[0, fcount] = int(f)
                fcount = fcount + 1
        else:
            x[0, fcount] = int(row[feature])
            fcount = fcount + 1
    y = float(row[value_field])
    return x, y

def biasStrength(rfr, mse, x, y):    
    predy = rfr.predict(x)
    if ( x[0,0] == 0 ):
        x[0,0] = 1
    else:
        x[0,0] = 0
    predy_oposite_gender = rfr.predict(x)
    print "Prediction:",predy
    print "Opposit gender prediction:",predy_oposite_gender
    err = (predy_oposite_gender - predy) ** 2
    bias = err/mse * 100
    return bias[0]

with open('joonTalk.pkl', 'rb') as input:
    data = pickle.load(input)
    rfr = data['rfr']
    datasetDict = data['datasetDict']
    num_features = data['num_features']
    value_field = data['value_field']
    features = data['features']
    mse = data['mse']

for i in [0, 1927, 8237, 20000]:
    row = datasetDict[i]
    print row,"\n"
    x, y = row2XY(row, features, num_features, value_field)
    bias = biasStrength(rfr, mse, x, y)
    print bias,"%"
    if ( bias > 50 ):
        print "Yes\n"
    else:
        print "No\n"

