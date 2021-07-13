import numpy as np
import pandas as pd
import shap
import sklearn as sk
import matplotlib.pyplot as plt
import scipy.stats as ss
import csv
import os.path as pt

#A/Define matrix of genotype-phenotype
class MMatrix:
    #1.Defining model variables and SHAP variables
    def __init__(self, param_g,param_N,param_M):
        #1.1.1.Size of matrices
        self.rndState=np.random.RandomState(10000)
        self.N = param_N #np.random.randint(1000, 5001)
        self.M = param_M #np.random.randint(10, 101)
        
        #1.1.2.Variance of coefficients(beta) and noise(epsilon)
        self.sigma_g_squared = param_g #np.random.uniform(0, 0.5)
        self.sigma_e_squared = 1 - self.sigma_g_squared
        
        #1.1.3.Distribution of coeeficients(beta) and noise(epsilon)
        self.beta = self.rndState.normal(0,self.sigma_g_squared/self.M,self.M)
        self.epsi = self.rndState.normal(0,self.sigma_e_squared,self.N)

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
                self.X[j][i] = self.rndState.binomial(2, p)

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
    
    #6.Correlation
    def mCorrelation(self,param_x1,param_x2):       
       #6.1.Person correlation
       self.pear_corr,self.pear_pval = ss.pearsonr(param_x1,param_x2)
       self.pear_corr = round(self.pear_corr,2)
       self.pear_pval = round(self.pear_pval,2)
       #if(self.pear_pval <=0.05): #Reject NULL: param_x1 == param_x2
       #    print("Pearson: corr="+str(self.pear_corr)+
       #          ", pval="+str(self.pear_pval))
       
       #6.2.Tau correlation
       self.tau_corr,self.tau_pval = ss.kendalltau(param_x1,param_x2)
       self.tau_corr = round(self.tau_corr,2)
       self.tau_pval = round(self.tau_pval,2)
       #if(self.tau_pval <=0.05): #Reject NULL: param_x1 == param_x2
       #    print("Tau: corr="+str(self.tau_corr)+
       #          ", pval="+str(self.tau_pval))

#B/Define stats operations
class MStat:
    #1.Define
    def __init__(self):
        self.pearson_corrs = ([])
        self.tau_corrs = ([])
        
        self.pearson_corr_mean = 0.0
        self.pearson_corr_sd = 0.0
        
        self.tau_corr_mean = 0.0
        self.tau_corr_sd = 0.0
        
    #3.Add data to list
    def addCorrelations(self,pearson_corr,tau_corr):
        self.pearson_corrs = np.append(self.pearson_corrs, pearson_corr)
        self.tau_corrs = np.append(self.tau_corrs, tau_corr)
    
    #2.Compare correlation values and get the standard deviation of this corr_ distribution
    def mCompareCorrSD(self):
        #7.1. Get mean of pearson correlations and see the standard deviation
        self.pearson_corr_mean = np.around(np.mean(self.pearson_corrs),2)
        self.pearson_corr_sd = np.around(np.std(self.pearson_corrs),2)
        #print("Pearson correlation: mean="+str(self.pearson_corr_mean)+
        #      ", standard deviation="+str(self.pearson_corr_sd))
        
        #7.1. Get mean of tau correlations and see the standard deviation
        self.tau_corr_mean = np.around(np.mean(self.tau_corrs),2)
        self.tau_corr_sd = np.around(np.std(self.tau_corrs),2)
        #print("Tau correlation: mean="+str(self.tau_corr_mean)+
        #      ", standard deviation="+str(self.tau_corr_sd))
            
    #4.Export data to file
    def exportFile(self, data, title):
        #5.1.1.Store data to daframe
        df = pd.DataFrame(data)
        #5.1.2.Store data frame into a .cvs file
        df.to_csv("output/"+title+".csv")
    
    #5.Export correlations data to files
    def exportCorrelations(self,title):
        #5.1.PEARSON: mean, std, data
        write_data = pd.DataFrame(np.transpose(np.vstack((self.pearson_corrs,
            self.tau_corrs))))
        
        pearson_data = pd.DataFrame(np.transpose([self.pearson_corr_mean,
                                                  self.pearson_corr_sd]))
        write_data = pd.concat([write_data.reset_index(drop=True),
                                pearson_data.reset_index(drop=True)],axis=1)
        
        #5.2.TAU:mean, std, data
        tau_data = pd.DataFrame(np.transpose([self.tau_corr_mean,
                                              self.tau_corr_sd]))
        write_data = pd.concat([write_data.reset_index(drop=True),
                                tau_data.reset_index(drop=True)],axis=1)
        
        #5.3.Naming column's headers
        write_data.columns = ["Pearson","Kendall Tau","Pearson Mean,Std",
                              "Kendall Tau Mean,Std"]
        
        self.exportFile(write_data, title)
        
#C/Loading area
if __name__ == "__main__":
    #Go through loops
    for i in range(1,2):#size NxM
        for j in range(1,11):#run 2 times 2 matrix of different g
            print("At g=",str(j/10))
            #Define the statatics object
            myStat = MStat()
            
            #And compute correlation values
            for k in range(0,100):
                #Define setting
                myMatrix = MMatrix(j/10,int((10**i)),int((10**i)/2))
                myMatrix.generateMatrix()
                myMatrix.simulatePhenotypeInf()
                myMatrix.generateModel(myMatrix.X, myMatrix.y, "infinitesimally")
                    
                #Export matrices
                """myMatrix.exportFile(myMatrix.X, "output/"+"["+str(myMatrix.N)+"x"+
                    str(myMatrix.M)+"]"+"genotype at g="+str(myMatrix.sigma_g_squared))
                myMatrix.exportFile(myMatrix.X, "output/"+"phenotype at g="+"["+
                    str(myMatrix.N)+"x"+str(myMatrix.M)+"]"+str(myMatrix.sigma_g_squared))
                    
                #Export charts
                myMatrix.exportComparisonChart(myMatrix.beta,"trueCoef",
                    myMatrix.sBeta,"predictedCoef",myMatrix.sABSValues,"meanedSHAPValues")
                """    
                #Correlation computation: Pearson, Tau
                myMatrix.mCorrelation(myMatrix.beta, myMatrix.sABSValues)
                    
                #Adding these correlations into computational places
                myStat.addCorrelations(myMatrix.pear_corr, myMatrix.tau_corr)
                        
                #Before output those values to screen
            myStat.mCompareCorrSD()
            myStat.exportCorrelations("correlations@["+str(myMatrix.N)+"x"+
                str(myMatrix.M)+"]"+"@g="+str(np.around(myMatrix.sigma_g_squared,2)))
            
