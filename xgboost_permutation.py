import numpy as np
import xgboost
from scipy import stats
import sklearn as sk
from sklearn.linear_model import LinearRegression
import shap
import os

num_permutations = 100

for phenotype_number in range (1, 201):
    print(str(phenotype_number) + "/200")
    cwd = os.getcwd()
    os.mkdir(cwd + '/xgboost_outputs/PermutationDistribution/' + str(phenotype_number))

    X = np.genfromtxt('outputs/0.1g/genotype_0.1g.csv', delimiter=',')
    y = np.genfromtxt('outputs/0.1g/phenotype_0.1g_' + str(phenotype_number) + '.csv', delimiter=',')
    
    for i in range (1, num_permutations+1):
        #fit a linear model to the genotype/phenotype data
        linear_model = sk.linear_model.LinearRegression()
        linear_model.fit(X, y)
    
        #get the phenotype predicted by the linear model
        y_hat = linear_model.predict(X)
    
        #estimate the residual by subtracting out the linear prediction from the true phenotype
        residual = np.subtract(y, y_hat)
    
        #permute the residual
        residual_prime = np.random.permutation(residual)
    
        #add the permuted residual back to the linear-effect phenotype prediction
        y_prime = y_hat + residual_prime
    
        #retrain xgboost model on y_prime
        Xd = xgboost.DMatrix(data=X,label=y_prime)
        xgboost_model = xgboost.train({'eta':0.3, 'max_depth':4, 'base_score': 0, "lambda": 0}, Xd, 1)
        
        #calculate shapley interaction values on the xgboost permutation model
        pred = xgboost_model.predict(Xd, output_margin=True)
        explainer = shap.TreeExplainer(xgboost_model)
        shap_values = explainer.shap_values(Xd)
    
        shap_interaction_values = explainer.shap_interaction_values(Xd)
    
        np.savetxt("xgboost_outputs/PermutationDistribution/" + str(phenotype_number) + "/XGBoost_" + str(i) + ".csv", shap_interaction_values[0], delimiter=",", fmt='%f')
    




