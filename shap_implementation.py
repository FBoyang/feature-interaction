import numpy as np
import random as rd
import pandas as pd
import shap
import sklearn as sk
import sympy as sp
import matplotlib.pyplot as plt

class MMatrix:
    #sample size (N) and genome length (M)
    N = int()
    M = int()
  
    maxGenotype = int()
    originalMatrix = []

    simulateInfPhenotype = []
    infPhenotypeContributionRateArray = np.array([])

    simulateUnInfPhenotype = []
    unInfPhenotypeContributionRateArray = np.array([])

    def __init__(self):
        #sample size (N) and genome length (M)
        self.N = rd.randint(1000, 5000)    #row
        self.M = 90        #column
        self.maxGenotype = 3               #{0,1,2}
        
    def generateMatrix(self):
        #generate matrix
        """
        self.originalMatrix = np.random.randint(
            self.maxGenotype, size=(self.N,self.M))
        """
        self.originalMatrix = np.random.multinomial(self.maxGenotype-1, 
            [1/self.N]*self.N, size =(self.N*self.M))
        
        
    def simulatePhenotypeInfinitesimally(self):
        print("Infinitesimal phenotype------------------------------------------")
        for i in range(self.M):
            self.infPhenotypeContributionRateArray = np.append(
                self.infPhenotypeContributionRateArray ,rd.uniform(-1, 1))

        transposedMatrix = self.infPhenotypeContributionRateArray.transpose()
        print(transposeMatrix)
        for i in range(self.N):
            self.simulateInfPhenotype = np.dot(
                self.originalMatrix,transposedMatrix)
        
    def simulatePhenotypeFinitesimally(self):
        print("Uninfinitesimal phenotype----------------------------------------")
        for i in range(self.M):
            if sp.isprime(i):
                self.unInfPhenotypeContributionRateArray = np.append(
                    self.unInfPhenotypeContributionRateArray,rd.uniform(-1, 1))
            else:
                self.unInfPhenotypeContributionRateArray = np.append(
                    self.unInfPhenotypeContributionRateArray,0.0)	
        
        transposedMatrix = self.unInfPhenotypeContributionRateArray.transpose()
        
        for i in range(self.N):
            self.simulateUnInfPhenotype = np.dot(
                self.originalMatrix,transposedMatrix)

    def generateModel(self, genotype, phenotype, modelType):
        print("Generating begun")
        X = pd.DataFrame(genotype)
        
        X.columns = ["G" + str(i) for i in range(self.M)]
        y = phenotype
        
        print("Model itself")
        model = sk.linear_model.LinearRegression()
        model.fit(X,y)
        
        #print("List all coefficients")
        #for i in range(X.shape[1]):
        #    print("[",i,"]", "=", model.coef_[i].round(4))
        
        #1. Partial dependence plotting
        #X100 = shap.utils.sample(X, 100)    
        #p=shap.plots.partial_dependence("G0", model.predict, X100, ice=False, 
        #    model_expected_value=True, feature_expected_value=True, show=False)
        
        #2.1. Compute SHAP values
        background = shap.maskers.Independent(X, max_samples=100)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X)
        #2.2. Make a standard partial dependence plot at a specific index
        sample_ind = 18
        #fig, ax = shap.plots.partial_depenedence_plot("G0", model.predict, X,
        #        model_expected_value = True, feature_expected_value = True,
        #        show=False,ice=False,shap_values=shap_values[sample_ind:
        #        sample_ind+1,:],shap_value_features=X.iloc[sample_ind:
        #        sample_ind+1,:] )
        #fig.show()
        
        #3. SHAP plot waterfall
        shap.plots.waterfall(shap_values[sample_ind], max_display=14)
        
        #plt.show(p)
        plt.savefig("X100@18" + modelType + ".png")
        
	
if __name__ == "__main__":
    """print(np.random.multinomial(2,[1/10.]*10,size=(100*30)))
    #"""
    print("New program=======================================================")    
    myMatrix = MMatrix()
    print("Generate matrix---------------------------------------------------")    
    myMatrix.generateMatrix()
    print("Generate phenotypes infinitesimally-------------------------------")
    myMatrix.simulatePhenotypeInfinitesimally()
    print("Generate phenotypes finitesimally---------------------------------")
    myMatrix.simulatePhenotypeFinitesimally()
    print("Generate model for phenotype infinitesimally----------------------")
    myMatrix.generateModel(myMatrix.originalMatrix, 
                           myMatrix.simulateInfPhenotype, "infinitesimally")
    print("Generate model for phenotype finitesimally------------------------")
    myMatrix.generateModel(myMatrix.originalMatrix, 
                           myMatrix.simulateUnInfPhenotype, "finitesimally")
    #"""