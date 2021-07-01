# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 10:52:07 2021

@author: JX
"""

import numpy as np
import random as rd
import pandas as pd
import shap
import sklearn as sk
import sympy as sp
import matplotlib.pyplot as plt
#import csv

class MMatrix:
    def __init__(self):
        #np.random.seed(1)
        self.N = 100#np.random.randint(1000, 5001)
        self.M = 50#np.random.randint(10, 101)

        self.infPhenotypeContributionRateArray = np.array([])
        self.unInfPhenotypeContributionRateArray = np.array([])
        
        self.shapCoefficience=np.empty(self.M) # predicted shap coeficiences
        self.maxDisplayInSHAP = 10 #wanna see maximum 10 features of SHAP
        self.sampleIndex = 1 #test SHAP correctioness at specific sample index
        self.maxSampleSize = 10 #to assign maximum sample size in shap.masker()
    
    def mNormalize(self, matrix):
        #return sk.preprocessing.normalize(matrix, axis=1, norm='l1')
        return matrix
        
    def generateMatrix(self):
        self.genotype_matrix = np.empty((self.N, self.M))

        for i, row in enumerate(self.genotype_matrix):
            np.random.seed(1)
            p = rd.uniform(0,1)
            for j, col in enumerate(row):
                self.genotype_matrix[i][j] = np.random.binomial(2, p)
                
            #normalized SNPs
            self.genotype_matrix = self.mNormalize(self.genotype_matrix)
        
    def simulatePhenotypeInfinitesimally(self):
        np.random.seed(1)
        for i in range(self.M):
            beta = rd.uniform(0,1)
            self.infPhenotypeContributionRateArray = np.append(
                self.infPhenotypeContributionRateArray ,beta)

        transposedMatrix = self.infPhenotypeContributionRateArray#.transpose()
        
        self.simulateInfPhenotype = np.dot(self.genotype_matrix,transposedMatrix)
        
    def simulatePhenotypeFinitesimally(self):
        for i in range(self.M):
            if sp.isprime(i):
                self.unInfPhenotypeContributionRateArray = np.append(
                    self.unInfPhenotypeContributionRateArray,rd.uniform(-1, 1))
            else:
                self.unInfPhenotypeContributionRateArray = np.append(
                    self.unInfPhenotypeContributionRateArray,0.0)

        transposedMatrix = self.unInfPhenotypeContributionRateArray.transpose()

        self.simulateUnInfPhenotype = np.dot(self.genotype_matrix,transposedMatrix)

    def generateModel(self, genotype, phenotype, modelType):
        X = pd.DataFrame(genotype)
        X.columns = ["G" + str(i) for i in range(self.M)]
        y = phenotype

        model = sk.linear_model.LinearRegression()
        model.fit(X,y)

        #Get all the coef. to an array for later comparision with true coef.
        for i in range(X.shape[1]):
            self.shapCoefficience[i] = format(model.coef_[i])
        
        #2.1. Compute SHAP values
        background = shap.maskers.Independent(X, max_samples=self.maxSampleSize)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X)
        
        #2.2. Make a standard partial dependence plot at a specific index
        sample_ind = self.sampleIndex
        
        #3. SHAP plot waterfall
        shap.plots.waterfall(shap_values[sample_ind], 
                             max_display=self.maxDisplayInSHAP)

    def exportPairDataToFile(self, data1, title1, data2, title2, fileName):
        df = pd.DataFrame(data1)
        df.columns = [title1]
        
        df[title2] = pd.DataFrame(data2)
        print(df)
        
        df.to_csv(fileName)

if __name__ == "__main__":
    myMatrix = MMatrix()
    myMatrix.generateMatrix()
    myMatrix.simulatePhenotypeInfinitesimally()
    myMatrix.simulatePhenotypeFinitesimally()
    myMatrix.generateModel(myMatrix.genotype_matrix,
                           myMatrix.simulateInfPhenotype, "infinitesimally")
    
    myMatrix.exportPairDataToFile(myMatrix.infPhenotypeContributionRateArray,
        'trueValue',myMatrix.shapCoefficience,'shapValue','./trueShapPairSamples.csv')
    
    """
    myMatrix.generateModel(myMatrix.genotype_matrix,
                           myMatrix.simulateUnInfPhenotype, "finitesimally")
    """
