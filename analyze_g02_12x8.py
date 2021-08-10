import numpy as np
import pandas as pd
import sklearn as sk
import csv
import os
import scipy as sp
import matplotlib.pyplot as plt

#Define analyze class
class MAnalyst:
    #Defining model variables and SHAP variables
    def __init__(self,param_g,param_N,param_K,param_P,param_I,param_J):
        self.K=param_K #total trials
        self.N=param_N #size of a trial
        self.P=param_P #permutation per trial
        self.G=param_g #sigma_g_squared
        
        #Pre-dddefine p-val
        self.prePVal = 0.05
       
        #Interaction between gene I and J
        self.I=param_I
        self.J=param_J
        
        #K trial,P permutation/trial of interaction scores
        self.pDistribution=np.empty((self.K,self.P))
       
        #pval along trials
        self.pPVal=np.empty(self.K)
    
        #Original interaction score along K trials
        self.originDistribution=np.empty(self.K)
        self.mPVal = 0.0
        self.mPower = 0.0
    
    #Get exactly data at position [i][j] of a csv file
    def getInteractScore(self,fileName):
        with open(fileName,'r') as f:
            my_csv=csv.reader(f)
            my_csv=list(my_csv)
            return my_csv[self.I-1][self.J-1]
    
    #Acquire data from analyze
    def acquireData(self):
        #Go through each csv file [k_idx][p_idx] and get interaction score (i,j)
        for k_idx in range(self.K):
            #Get original interaction score at (i,j) position
            fileName="InteractionScore/g@"+str(self.G)+"_12x8_S/IS_NN_"+str(k_idx+1)+'.csv'
            self.originDistribution[k_idx]=self.getInteractScore(fileName)
            
            #Get permutated interaction scores at (i,j) poaition
            for p_idx in range(self.P):
                #Locate the stored files - f
                f="PermutationDistribution/g@"+str(self.G)+"_12x8_S_2/PIS_trial@"+str(k_idx+1)+'_NN_permuted@'+str(p_idx+1)+'.csv'
                self.pDistribution[k_idx][p_idx]=self.getInteractScore(f)

    #Export gene to file
    def exportFile(self, data, title):
        #Store data frame into a .cvs file
        np.savetxt(title+'.csv',data,delimiter=",")
        
    #Compute p-value per trial
    def computePVal(self):
        #Storing percentile value for showing later
        mPercentile = np.array([])
        
        #All trials, how many permutated data is out of scope of original data
        total_out_of_scope = 0
        for k_idx in range(self.K):
            #Per trial: How many permutated data is out of scope of original data
            out_of_expect=0
            for p_idx in range(self.P):
                #times of out of scope
                if self.pDistribution[k_idx][p_idx]>self.originDistribution[k_idx]:
                    out_of_expect +=1
            #Percentage of this trial’s observations
            self.pPVal[k_idx]=out_of_expect/self.P
            
            #Counting all trials
            if self.pPVal[k_idx] < self.prePVal:
                total_out_of_scope +=1
            
            #For plotting purpose later
            mPercentile=np.append(mPercentile,sp.stats.percentileofscore(
                self.pDistribution[k_idx], self.originDistribution[k_idx]))
            
        #Answer the first question of the function now
        self.mPVal = total_out_of_scope/self.K
        
        plt.hist(mPercentile,bins=100)
        plt.xlabel('percentages')
        plt.ylabel('frequency')
        
        plt.savefig('Summary@'+str(self.G)+'_12x8_FPR@'+str(self.mPVal)+'.jpg')
        
    #Compute power
    def computePower(self):
        #Storing percentile value for showing later
        mPercentile = np.array([])
        
        #All trials, how many permutated data is out of scope of original data
        total_in_of_scope = 0
        for k_idx in range(self.K):
            #Per trial: How many permutated data is out of scope of original data
            in_of_expect=0
            for p_idx in range(self.P):
                #times of in of scope
                if self.pDistribution[k_idx][p_idx]<=self.originDistribution[k_idx]:
                    in_of_expect +=1
            #Percentage of this trial’s observations
            self.pPVal[k_idx]=in_of_expect/self.P
            
            #Counting all trials
            if self.pPVal[k_idx] <= self.prePVal:
                total_in_of_scope +=1
            
            #For plotting purpose later
            mPercentile=np.append(mPercentile,sp.stats.percentileofscore(
                self.pDistribution[k_idx], self.originDistribution[k_idx]))
            
        #Answer the first question of the function now
        self.mPower = 1-total_in_of_scope/self.K
        
        plt.hist(mPercentile,bins=100)
        plt.xlabel('percentages')
        plt.ylabel('frequency')
        
        plt.savefig('Summary_Power@'+str(self.G)+'_12x8@'+str(self.mPower)+'.jpg')
        
#Loading area
if __name__ == "__main__":
    N=1024 #Total individiuals
    M=20 #Total SNPs of each individual
    K=200 #Total set of {N individuals x M SNPs} //{NxM}
    sigma_g=0.02 #Variance of noise to simulate phenotype

    mAnalyst = MAnalyst(sigma_g,N,K,permutationTimes,1,2)
    mAnalyst.acquireData()
    mAnalyst.computePVal()
    mAnalyst.computePower()
