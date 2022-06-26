from types import coroutine
import numpy as np
from pyparsing import alphas
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline


data = {}

list_cases = ["worst", "best", "average"]
for assumption in list_cases:
    # Uses randomized data with random chosen methodologies
    data[assumption] = pd.read_csv("ADEME_uncertainty_assessement/ADEME_uncertainty_assessment_"+assumption+"_random"+".csv")

def get_linear_regression_results():
    list_data = []
    for assumption in list_cases:
        data_to_use = data[assumption]
        data_to_use["assumption"] = assumption
        list_data.append(data_to_use)

    data_to_use = pd.concat(list_data)
    data_to_use = data_to_use.dropna(axis=1, how="all")
    data_to_use = data_to_use.dropna(axis=0, how="any")
    
    list_coef=[]

    
    for i in range(1,24):
        if i == 9:
            i = "9a"
        if i == 8 or i == 18 or i == 19 or i == 20 or i == 23:
            continue
        data_to_use['variance '+str(i)] = (data_to_use["Total uncertainty "+str(i)]*data_to_use["mean "+str(i)])


    data_to_use.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_to_use = data_to_use.dropna(axis=0, how='any')

    X1 = data_to_use[[column for column in data_to_use.columns if "standard" in column]].values.tolist()
    X2 = data_to_use[[column for column in data_to_use.columns if "mean" in column]].values.tolist()
    for i in range(0, len(X1)):
        X1[i] = X1[i]+X2[i]
    X_train = X1[1:-100]
    X_test = X1[-100:]

    Y = data_to_use[['Most probable total value (tCO2e/collaborator)']].values.tolist()
    Y_train = Y[1:-100]
    Y_test = Y[-100:]

    # Plots a correlation plot to check dependencies
    corr_data = data_to_use[['Most probable total value (tCO2e/collaborator)']+["Interval lenght"]+['Uncertainty measure']]
    corr_data['Sum of variances'] =  data_to_use[[column for column in data_to_use.columns if "variance" in column]].astype(float).sum(axis=1)
    corr_data["assumption"] = data_to_use["assumption"]

    corr_data.rename({'Most probable total value (tCO2e/collaborator)': "Median total emission \n (tCO2e/collaborator)", "assumption":"Data quality level"}, axis=1,inplace=True)
    corr_data.reset_index(drop=True, inplace=True)
    
    ax  = sn.pairplot(corr_data, hue="Data quality level", kind = "kde")
    plt.show()

    plt.figure()

    # Plots actual values against predicted value
    plt.plot(Y_test, label="Data")

    for alpha in np.arange(0,100,1):
        model = Lasso(alpha=alpha)

        model.fit(X_train,Y_train)

        Y_pred = model.predict(X_test)
        print(Y_test, Y_pred)
        
        plt.plot(Y_pred, label="Prediction "+str(alpha))
        print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
        print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred, force_finite=False))
        list_coef.append(r2_score(Y_test, Y_pred, force_finite=False))
        print(model.coef_)

    plt.legend()
    plt.show()

    # Plots the mean squared error against the Lasso parameter to look for an optimum
    plt.figure()
    plt.plot(np.arange(0,100,1), list_coef)
    plt.show()
    print(list_coef)


get_linear_regression_results()

    

