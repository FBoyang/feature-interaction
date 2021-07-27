import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import scipy.stats as ss
import csv
import os
#import run_interaction as ri
#import run_permutation as rp
#from datetime import datetime

#Define matrix of genotype-phenotype
class MMatrix:
    #Defining model variables and SHAP variables
    def __init__(self, param_g,param_N,param_M):
        #1.1.1.Size of matrices
        self.rndState=np.random.RandomState(10000)
        self.N = param_N
        self.M = param_M
        
        #1.1.2.This simulation generate phenotype with ONLY noise
        self.sigma_g_squared = param_g
        self.sigma_e_squared = 1 - param_g
        
        
        #1.1.3.Distribution of coeeficients(beta) and noise(epsilon)
        self.beta = np.empty(self.M)
        self.epsi = np.empty(self.N)

        #1.1.4.Original model
        self.X = np.empty((self.N, self.M)) # Gene matrix X (N*M)
        self.y = np.empty(self.N) # original Phenotype matrix y (N)
    
    #Generate a genotype matrix
    def generateGenotype(self):
        #self.rndState=np.random.RandomState(10000)
        #2.1.Generate genotype matrix X
        for i in range(self.M):
            p = np.random.uniform(0, 1)
            for j in range(self.N):
                self.X[j][i] = np.random.binomial(2, p)

        #2.2.Normalize the genotype matrix column-wisely
        #self.X = sk.preprocessing.scale(self.X)
    
    #Simulate a phenotype matrix infinitesimally
    def simulatePhenotypeInf(self):
        #Distribution of coeeficients(beta) and noise(epsilon)
        self.beta = self.rndState.normal(0,self.sigma_g_squared/self.M,self.M)
        self.epsi = self.rndState.normal(0,self.sigma_e_squared,self.N)
        
        #NOW y = e
        self.y = np.add(np.dot(self.X,self.beta.transpose()),self.epsi)
    
    #Export gene to file
    def exportFile(self, data, directory, title):
        #Check and create corresponding dirs
        if not os.path.exists(directory):
            os.makedirs(directory)
        #Store data frame into a .cvs file
        np.savetxt(title+'.csv',data,delimiter=",")
     
#Loading area
if __name__ == "__main__":
    N=1024 #Total individiuals
    M=20 #Total SNPs of each individual
    K=200 #Total set of {N individuals x M SNPs} //{NxM}
    sigma_g=0.02 #Variance of noise to simulate phenotype
    permutationTimes=50 #Total permutation times per original set {NxM}
    
    #Generate K X,y and export it to K output files {X,y}
    totalRunningTime = []
    fXData=""
    fyData=""
    
    myMatrix = MMatrix(sigma_g,N,M)
    myMatrix.generateGenotype()
    
    myDirectory = "data/g@"+str(sigma_g)
    myTitle = myDirectory + "/genotype"
    myMatrix.exportFile(myMatrix.X,myDirectory,myTitle)
    
    for i in range(K):
        #Define setting
        myMatrix.simulatePhenotypeInf()
                    
        #Export genotype-phenotype at ith position
        myDirectory = "data/g@"+str(sigma_g)
        myTitle = myDirectory + "/phenotype_"+str(i+1)
        myMatrix.exportFile(myMatrix.y,myDirectory,myTitle)
    print("Done generate genome")