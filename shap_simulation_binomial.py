import numpy as np
import random as rd
import pandas as pd
import shap
import sklearn as sk
import sympy as sp
import matplotlib.pyplot as plt

#A/Defining class
class MMatrix:
    #1.Defining model variables and SHAP variables
    def __init__(self):
        #1.1.1.Size of matrices
        np.random.seed(1)
        self.N = np.random.randint(1000, 5001)
        self.M = np.random.randint(10, 101)
        
        #1.1.2.Variance of coefficients(beta) and noise(epsilon)
        self.sigma_e_squared = rd.uniform(0,0.5)
        self.sigma_g_squared = 1-self.sigma_e_squared
        
        #1.1.3.Distribution of coeeficients(beta) and noise(epsilon)
        self.beta = np.random.normal(0,self.sigma_g_squared/self.M,self.M)
        self.epsi = np.random.normal(0,self.sigma_e_squared,self.N)
        
        #1.1.4.Original model
        self.X = np.empty((self.N, self.M)) # Gene matrix X (N*M)
        self.y = np.empty(self.N) # original Phenotype matrix y (N)
                
        #1.2.1.SHAP modeling
        self.sBeta = np.empty(self.M) # predicted shap coeficiences
        self.sy = np.empty(self.N) # predicted shap phenotype
        
        #1.2.2.SHAP parameters
        self.maxDisplayInSHAP = 10 #wanna see maximum 10 features of SHAP
        self.sampleIndex = 1 #test SHAP correctioness at specific sample index
        self.maxSampleSize = 10 #to assign maximum sample size in shap.masker()
        
    def mNormalize(self, matrix):
        #2.1. (1)-D matrix
        if (len(matrix)==1):
            norm = np.linalg.norm(matrix)
            normal_array = matrix/norm
            return normal_array
        #2.2. (2+)-D matrix
        else:
            return sk.preprocessing.normalize(matrix, axis=1, norm='l1')
        
    #2.Generate a genotype matrix
    def generateMatrix(self):
        #2.1.Generate genotype matrix X
        for i, col in enumerate(self.X):
            np.random.seed(1)
            p = rd.uniform(0,1)
            for j, row in enumerate(col):
                self.X[i][j] = np.random.binomial(2, p)
    
    #3.Simulate a phenotype matrix infinitesimally
    def simulatePhenotypeInf(self):
        self.y = np.add(np.dot(self.X,self.beta),self.epsi)

    #4.Fitting X,y to linear model and SHAP-ing the model
    def generateModel(self, genotype, phenotype, modelType):
        #4.1.Naming each gene in the genotype matrix by SNP orders
        X = pd.DataFrame(genotype)
        X.columns = ["G" + str(i) for i in range(self.M)]
        y = phenotype
        
        #4.2.1.Fitting X,y to model
        model = sk.linear_model.LinearRegression()
        model.fit(X,y)

        #4.2.2.Get all the coef. to an array for later comapre with true coef.
        for i in range(X.shape[1]):
            self.sBeta[i] = format(model.coef_[i])
        
        #4.2.3.Compute SHAP values
        background = shap.maskers.Independent(X, max_samples=self.maxSampleSize)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X)
        
        #4.2.4.Get all the phenotype to an array for later compare with true pheno.
        for i in range(self.N):
            self.sy[i] = shap_values.base_values[i]
            for j in range(self.M):
                self.sy[i] +=self.sBeta[i]*shap_values.data[i][j]
        
        #4.3.1.Make a standard partial dependence plot at a specific index
        sample_ind = self.sampleIndex
        
        #4.3.2.SHAP plot waterfall
        shap.plots.waterfall(shap_values[sample_ind],
                             max_display=self.maxDisplayInSHAP, show=False)
        
        #4.3.3.Save result to a file
        plt.savefig("waterfall"+modelType+".png")
    
    #5.Export data to a file to run R analysis
    def exportPairDataToFile(self, data1, title1, data2, title2, fileName):
        #5.1.Store first column of data
        df = pd.DataFrame(data1)
        df.columns = [title1]
        #5.2.Store second column of data
        df[title2] = pd.DataFrame(data2)
        print(df)
        #5.3.Store data frame into a .cvs file
        df.to_csv(fileName)
        
    #6.Export gene data to 2 files
    def exportGeneDataToFile(self, data1, fileName1, data2, fileName2):
        #6.1.Store gene data to file 1
        df = pd.DataFrame(data1)
        print("X\n",df)
        #5.2.Store data frame into a .cvs file
        df.to_csv(fileName1)
        
        #6.1.Store gene data to file 2
        df = pd.DataFrame(data2)
        print("y\n",df)
        #5.2.Store data frame into a .cvs file
        df.to_csv(fileName2)

#B.Executing the defined class
if __name__ == "__main__":
    #1.Calling matrix object
    myMatrix = MMatrix()
    
    #2.Generate corresponding matrices
    myMatrix.generateMatrix()
    
    #3.Simulate phenotype matrix infinitesimally
    myMatrix.simulatePhenotypeInf()
    
    #4.Fitting dataset (X,y) to linear model and observing result
    myMatrix.generateModel(myMatrix.X, myMatrix.y, "_infinitesimally")
    
    #5.Store paired data of coef_ in both original and SHAP-ed model to file
    myMatrix.exportPairDataToFile(myMatrix.beta,'trueValue',
            myMatrix.sBeta,'shapValue','./trueShapPairSamples.csv')
    
    #6.Store geno- and pheno- data into files
    myMatrix.exportGeneDataToFile(myMatrix.X,'genotype.csv',
                                  myMatrix.y,'phenotype.csv')
    #6.Store paired data of phenotype of oriniginal and SHAP-ed model to file
    #myMatrix.exportPairDataToFile(myMatrix.y,
    #    'trueValue',myMatrix.sy,'shapValue','./trueShapPairSamples.csv')
