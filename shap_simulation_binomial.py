import numpy as np
import pandas as pd
import shap
import sklearn as sk
import sympy as sp
import matplotlib.pyplot as plt


class MMatrix:
    def __init__(self):
        np.random.seed(1)
        self.N = np.random.randint(1000, 5001)
        self.M = np.random.randint(10, 101)

        self.infPhenotypeContributionRateArray = np.array([])
        self.unInfPhenotypeContributionRateArray = np.array([])

    def generateMatrix(self):
        self.genotype_matrix = np.empty((self.N, self.M))

        for i, row in enumerate(self.genotype_matrix):
            p = np.random.random()
            for j, col in enumerate(row):
                self.genotype_matrix[i][j] = np.random.binomial(2, p)

    def simulatePhenotypeInfinitesimally(self):
        for i in range(self.M):
            self.infPhenotypeContributionRateArray = np.append(
                self.infPhenotypeContributionRateArray ,np.random.uniform(-1, 1))

        transposedMatrix = self.infPhenotypeContributionRateArray.transpose()

        self.simulateInfPhenotype = np.dot(self.genotype_matrix,transposedMatrix)

    def simulatePhenotypeFinitesimally(self):
        for i in range(self.M):
            if sp.isprime(i):
                self.unInfPhenotypeContributionRateArray = np.append(
                    self.unInfPhenotypeContributionRateArray,np.random.uniform(-1, 1))
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
        shap.plots.waterfall(shap_values[sample_ind], max_display=14) #add show=false for plt.savefig to work

        #plt.show(p)
        plt.savefig("outputs/" + modelType + ".png")


if __name__ == "__main__":
    myMatrix = MMatrix()
    myMatrix.generateMatrix()
    myMatrix.simulatePhenotypeInfinitesimally()
    myMatrix.simulatePhenotypeFinitesimally()
    myMatrix.generateModel(myMatrix.genotype_matrix,myMatrix.simulateInfPhenotype, "infinitesimally")
    myMatrix.generateModel(myMatrix.genotype_matrix,myMatrix.simulateUnInfPhenotype, "finitesimally")
