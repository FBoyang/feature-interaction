import numpy as np
import pandas as pd
import shap
import sklearn as sk
import sympy as sp
import matplotlib.pyplot as plt

class MMatrix:
    #1.Defining model variables and SHAP variables
    def __init__(self, param_g,param_N,param_M):
        #1.1.1.Size of matrices
        np.random.seed(10000)
        self.N = param_N #np.random.randint(1000, 5001)
        self.M = param_M #np.random.randint(10, 101)
        
        #1.1.2.Variance of coefficients(beta) and noise(epsilon)
        self.sigma_g_squared = param_g #np.random.uniform(0, 0.5)
        self.sigma_e_squared = 1 - self.sigma_g_squared
        
        #1.1.3.Distribution of coeeficients(beta) and noise(epsilon)
        self.beta = np.random.normal(0,self.sigma_g_squared/self.M,self.M)
        self.epsi = np.random.normal(0,self.sigma_e_squared,self.N)

        #1.1.4.Original model
        self.X = np.empty((self.N, self.M)) # Gene matrix X (N*M)
        self.y = np.empty(self.N) # original Phenotype matrix y (N)

        #1.2.1.SHAP modeling
        self.sBeta = np.empty(self.M) # predicted shap coeficiences
        self.sABSValues = np.empty(self.M) # neutralized absoluted SHAP values
        
        #1.2.2.SHAP parameters
        self.maxDisplayInSHAP = 10 #wanna see maximum 10 features of SHAP
        self.sampleIndex = 1 #test SHAP correctioness at specific sample index
        self.maxSampleSize = 10 #to assign maximum sample size in shap.masker()

    #2.Generate a genotype matrix
    #Credit Henry,Nick    
    def generateMatrix(self):
        #2.1.Generate genotype matrix X
        for i in range(self.M):
            p = np.random.uniform(0, 1)
            for j in range(self.N):
                self.X[j][i] = np.random.binomial(2, p)

        #2.2.Normalize the genotype matrix column-wisely
        self.X = sk.preprocessing.scale(self.X)
    
    #3.Simulate a phenotype matrix infinitesimally
    def simulatePhenotypeInf(self):
        #y = X.b + e
        self.y = np.add(np.dot(self.X,self.beta.transpose()),self.epsi)
    
    #4.Fitting X,y to linear model and SHAP-ing the model
    def generateModel(self, genotype, phenotype, modelType):
        #4.1.1.Naming each gene in the genotype matrix by SNP orders
        X = pd.DataFrame(genotype)
        #4.1.2.Naming each SNP
        X.columns = ["SNP" + str(i+1) for i in range(self.M)]
        y = phenotype
        
        #4.2.1.Fitting X,y to model
        model = sk.linear_model.LinearRegression()
        model.fit(X,y)
        
        #4.2.2.Get all the coef. to an array for later comapre with true coef.
        self.sBeta = model.coef_
        
        #4.2.3.Compute SHAP values
        background = shap.maskers.Independent(X, self.maxSampleSize)
        explainer = shap.Explainer(model=model.predict, masker=background)
        shap_values = explainer(X)

        #4.2.4.Neutralized absoluted SHAP values with mean at 0
        self.sABSValues = np.abs(shap_values.values).mean(0)
    
    #5.1.Export gene to file
    def exportFile(self, data, title):
        #5.1.1.Store data to daframe
        df = pd.DataFrame(data)
        #5.1.2.Store data frame into a .cvs file
        df.to_csv(title+".csv")
    
    #5.2.Export {trueCoef,sklearnCoef,shapValues} comparison to charts
    #Credit: Nick
    def exportComparisonChart(self,data1,title1,data2,title2,data3,title3):
        #5.2.1.Collect data in orders
        mainframe = np.transpose(np.vstack((data1,data2,data3)))
        self.exportFile(mainframe,"output/"+"["+str(self.N)+"x"+str(self.M)+
            "]"+title1+"_"+title2+"_"+title3+" at g="+str(self.sigma_g_squared))
        
        #5.2.2.Prepare data for charting
        mainframe = np.transpose(np.vstack((np.abs(data1),np.abs(data2),np.abs(data3))))
        header = [title1,title2,title3]
        
        #5.2.3.Set up scale inside the chart
        x =np.arange(mainframe.shape[0]) #how many bins to show on plot
        
        dx=np.arange(mainframe.shape[1])-mainframe.shape[1]/2 #positions of each column in a bin
        dx /= mainframe.shape[1]+1 #neutralize as relative positions
        d = 1/(mainframe.shape[1]+1)#width of each column in a bin
        
        #5.2.4.Plotting task
        fig, ax = plt.subplots()
        for i in range(mainframe.shape[1]):
            ax.bar(x+dx[i], mainframe[:,i], width=d, label=header[i])
        
        #5.2.5.Store chart
        plt.legend(framealpha=1)
        plt.savefig("comparison/"+"["+str(self.N)+"x"+str(self.M)+"]"+
                    str(self.sigma_g_squared) + "g.png", dpi=1000)
        plt.close()
    
if __name__ == "__main__":
    for i in range(0,11):
        #Define setting
        myMatrix = MMatrix(i/10,100,70)
        myMatrix.generateMatrix()
        myMatrix.simulatePhenotypeInf()
        myMatrix.generateModel(myMatrix.X, myMatrix.y, "infinitesimally")
        #EWxport matrices
        myMatrix.exportFile(myMatrix.X, "output/"+"["+str(myMatrix.N)+"x"+
            str(myMatrix.M)+"]"+"genotype at g="+str(myMatrix.sigma_g_squared))
        myMatrix.exportFile(myMatrix.X, "output/"+"phenotype at g="+"["+
            str(myMatrix.N)+"x"+str(myMatrix.M)+"]"+str(myMatrix.sigma_g_squared))
        #Export charts
        myMatrix.exportComparisonChart(myMatrix.beta,"trueCoef",
            myMatrix.sBeta,"predictedCoef",myMatrix.sABSValues,"meanedSHAPValues")
