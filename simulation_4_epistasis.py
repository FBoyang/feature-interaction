import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import scipy.stats as ss
import csv
import os

#Define matrix of genotype-phenotype
class MMatrix:
    #Defining model variables and SHAP variables
    def __init__(self, pGenes, pSizeOfX, pSigG, pSN, pN,pEffects):
        #1.1.1.Size of matrices
        self.N = pN                         # number of individuals
        self.geneNum = pGenes               # number of genes (genes)
        self.sizeOfGene = pSizeOfX          # size of a gene (SNPs)
        self.M = pGenes * pSizeOfX          # size of an individual (SNPs)
        self.effSize = pEffects             # size of gene-gene-effect
        self.effName = "_equal_add_mul_s2a" # add, multiply, square 2nd then add
        
        #1.1.2. Setting of distribution variance/coefficient
        self.sig_g = pSigG                  # main-effect variance
        self.snr = pSN                      # signal to noise ratio
        self.sig_e = 1/(1+self.snr)         # --> noise-effect variance
        self.sig_gg = 1-pSigG-1.0/(pSN+1)   # --> gene-gene-effect variance
        
        #1.1.3.Distribution: 
        #   effect coefficients{(gene:alpha),(main:beta),(gene-gene:gamma)}
        #   noise {(noise:epsilon)}
        self.alpha=np.empty((self.geneNum,self.sizeOfGene)) #gene coefficient(pGenes*Size)
        self.beta =np.empty(self.geneNum)                   #main coefficient(pGenes)
        self.gamma=np.empty(self.effSize)                   #gene-gene-effect coefficient
        self.epsi =np.empty(self.N)                         #noise effect
        
        #1.1.4.Original model
        self.X = np.empty((self.N, self.M)) # Gene matrix X (N*M)
        self.uX= np.empty(((self.geneNum,self.N, self.sizeOfGene))) # un-weighted gene
        self.wX= np.empty((self.N,self.geneNum)) # weighted gene
        self.y = np.empty(self.N) # original Phenotype matrix y (N)

    #Generate a genotype matrix
    def generateGenotype(self):
        #2.1.Genotype generation
        for i in range(self.geneNum):
            #2.1.1.Each gene has its own normal distribution
            self.rndState=np.random.RandomState(20000)
            for j in range(self.sizeOfGene):
                p = np.random.uniform(0, 1)
                for k in range(self.N):
                    self.X[k][i*self.sizeOfGene+j] = np.random.binomial(2,p)
                    #get the unweighted genotype also
                    self.uX[i][k][j] = self.X[k][i*self.sizeOfGene+j]
        
        #2.3.Generate SNP size files
        #2.3.1.Working directory data/g@g_S/N_2_10/
        mDir="data/g@"+str(self.sig_g)+"_"+str(self.snr)
        mDir+="_"+str(self.geneNum)+"_"+str(self.sizeOfGene)+self.effName
              
        #2.3.2.Working file data/g@g_S/N_2_10/snps_size.csv
        mSNPTitle=mDir+"/snps_size"
        geneStructure = [self.sizeOfGene for i in range(self.geneNum)]
        self.exportFile(geneStructure,mDir,mSNPTitle)
              
        #2.3.3.Working file data/g@g_S/N_2_10/genotype.csv
        mGeneTitle=mDir+"/genotype"
        self.exportFile(self.X,mDir,mGeneTitle)
        
    #Simulate a phenotype matrix finitesimally
    def simulatePhenotypeF(self):
        #3.1.1.Distribution of gene coefficients(alpha)
        self.rndState=np.random.RandomState(20000)
        self.wX = self.wX.transpose()
        for i in range(self.geneNum):
            self.alpha[i] = np.random.normal(0,1/self.sizeOfGene,self.sizeOfGene)
      
            #3.1.2.Let's convert un-weighted to weighted gene first: 
                ###g_i=alpha_i*X_i###
            self.wX[i] = np.dot(self.uX[i],self.alpha[i])
    
        #3.2.1.Distribution of main effect
        self.beta = np.random.normal(0,self.sig_g/self.geneNum,self.geneNum)
        
        #3.2.2.Distribution of different gene-gene effect
        self.gamma = np.random.normal(0,self.sig_gg/self.effSize,self.effSize)
        
        #3.2.4.Distribution of noise effect
        self.epsi = np.random.normal(0,self.sig_e,self.N)

        #3.3.Construct phenotype:
        ###g_i*beta+eff_1*(g_i+g_j)+eff_2*(g_i-g_j)+eff_3*max(g_i,g_j)+e###
        self.main  = np.dot(self.wX.transpose(),self.beta)
        
        #Suggestive operations
        self.ggAdd = (np.add(self.wX[0],self.wX[1])*self.gamma[0]).transpose()
        self.ggMul = ((self.wX[0]*self.wX[1])*self.gamma[1]).transpose()
        self.ggS2A = (np.add(self.wX[0],self.wX[1]*self.wX[1])*self.gamma[2]).transpose()
        #Square 2nd term and add
        
        #Optional operations
        #self.ggMax = (np.maximum(self.wX[0],self.wX[1])*self.gamma[0]).transpose()
        #self.ggSub = (np.subtract(self.wX[0],self.wX[1])*self.gamma[1]).transpose()
        #self.ggMin = (np.minimum(self.wX[0],self.wX[1])*self.gamma[2]).transpose()
        
        self.wX = self.wX.transpose() #back to original form
    
        self.y = np.add(self.main,self.ggAdd,self.ggMul)
        self.y = np.add(self.y,self.ggS2A,self.epsi)
      
    #Export gene to file
    def exportFile(self, data, directory, title):
        #Check and create corresponding dirs
        if not os.path.exists(directory):
            os.makedirs(directory)
        #Store data frame into a .cvs file
        np.savetxt(title+'.csv',data,delimiter=",",fmt='%f')

#Loading area
if __name__ == "__main__":
    simulationTime=200
    #Initialize the geno-pheno setting    
    for pSig in [0.0,0.1,0.2,0.6,1.0]:#[0.0,0.1,0.2,0.6,0.8,1.0]:
        for pStoN in [1.0]:#[1.0,0.5,0.15,0.1,0.07]:#real data insights
            mMat=MMatrix(pGenes=2,pSizeOfX=10,pSigG=pSig,pSN=pStoN,pN=10240,pEffects=3)
            #Generate genotype
            mMat.generateGenotype()
            for i in range(simulationTime):
                mMat.simulatePhenotypeF()
                
                #2.3.1.Working directory data/g@g_S/N_2_10/
                mDir="data/g@"+str(mMat.sig_g)+"_"+str(mMat.snr)
                mDir+="_"+str(mMat.geneNum)+"_"+str(mMat.sizeOfGene)+mMat.effName
              
                #2.3.2.Working file data/g@g_S/N_2_10/snps_size.csv
                mPhenotypeTitle=mDir+"/phenotype_"+str(i+1)
                mMat.exportFile(mMat.y,mDir,mPhenotypeTitle)
                
    print("Done generate genome")
