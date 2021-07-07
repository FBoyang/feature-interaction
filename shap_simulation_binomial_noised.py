import numpy as np
import pandas as pd
import shap
import sklearn as sk
import sympy as sp
import matplotlib.pyplot as plt


class MMatrix:
    def __init__(self, g):
        np.random.seed(12345)
        self.N = 5000 #np.random.randint(1000, 5001)
        self.M = 100 #np.random.randint(10, 101)

        self.inf_beta_array = np.array([])
        self.uninf_beta_array = np.array([])

        self.inf_epsilon_array = np.array([])
        self.uninf_epsilon_array = np.array([])

        self.sigma_g_squared = g #np.random.uniform(0, 0.5)
        self.sigma_e_squared = 1 - self.sigma_g_squared

        self.g = str(np.around(g, decimals=2))

    def generateMatrix(self):
        self.genotype_matrix = np.empty((self.N, self.M))

        for i in range(0, self.M):
            p = np.random.uniform(0, 1) #p is fixed column-wise
            for j in range(0, self.N):
                self.genotype_matrix[j][i] = np.random.binomial(2, p)

    def simulatePhenotypeInfinitesimally(self):
        for i in range(self.M):
            self.inf_beta_array = np.append(self.inf_beta_array, np.random.normal(0, self.sigma_g_squared/self.M))

        transposed_matrix = self.inf_beta_array.transpose()

        #pre-noise simulated phenotype
        self.simulate_inf_phenotype = np.dot(self.genotype_matrix, transposed_matrix)

        #populate epsilon (noise) array)
        for i in range(self.N):
            self.inf_epsilon_array = np.append(self.inf_epsilon_array, np.random.normal(0, self.sigma_e_squared))

        #post-noise simualted phenotype
        self.simulate_inf_phenotype_noised = np.add(self.simulate_inf_phenotype, self.inf_epsilon_array)

    # def simulatePhenotypeFinitesimally(self):
    #     for i in range(self.M):
    #         if sp.isprime(i): #the isprime condition can be changed to whatever SNPs you want to be included
    #             self.uninf_beta_array = np.append(self.uninf_beta_array, np.random.normal(0, self.sigma_g_squared/self.M))
    #         else:
    #             self.uninf_beta_array = np.append(self.uninf_beta_array, 0.0)
    #
    #     transposed_matrix = self.uninf_beta_array.transpose()
    #
    #     #pre-noise simulated phenotype
    #     self.simulate_uninf_phenotype = np.dot(self.genotype_matrix, transposed_matrix)
    #
    #     #populate epsilon (noise) array)
    #     for i in range(self.N):
    #         self.uninf_epsilon_array = np.append(self.uninf_epsilon_array, np.random.normal(0, self.sigma_e_squared))
    #
    #     #post-noise simualted phenotype
    #     self.simulate_uninf_phenotype_noised = np.add(self.simulate_uninf_phenotype, self.uninf_epsilon_array)

    def generateModel(self, genotype, phenotype, modelType):
        X = pd.DataFrame(genotype)

        X.columns = ["G" + str(i+1) for i in range(self.M)]
        y = phenotype

        model = sk.linear_model.LinearRegression()
        model.fit(X,y)

        # print("Model coefficients for " + modelType + ":")
        # for i in range(X.shape[1]):
        #    print("[G" + str(i+1) + "]", "=", model.coef_[i].round(4))

        #1. Partial dependence plotting
        # X100 = shap.utils.sample(X, 100)
        # p=shap.plots.partial_dependence("G0", model.predict, X100, ice=False,
        #    model_expected_value=True, feature_expected_value=True, show=False)

        #2.1. Compute SHAP values
        background = shap.maskers.Independent(X, max_samples=100)
        explainer = shap.Explainer(model=model, masker=background)
        shap_values = explainer(X)

        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.savefig("outputs/" + modelType + "_summary_" + self.g + "g" + ".png")
        plt.close()

        #2.2. Make a standard partial dependence plot at a specific index
        sample_ind = 18
        # fig, ax = shap.plots.partial_depenedence_plot("G0", model.predict, X,
        #        model_expected_value = True, feature_expected_value = True,
        #        show=False,ice=False,shap_values=shap_values[sample_ind:
        #        sample_ind+1,:],shap_value_features=X.iloc[sample_ind:
        #        sample_ind+1,:] )
        # fig.show()

        #3. SHAP plot waterfall
        shap.plots.waterfall(shap_values[sample_ind], max_display=20, show=False)
        plt.savefig("outputs/" + modelType + "_waterfall_" + self.g + "g" + ".png")
        plt.close()


if __name__ == "__main__":
    for i in range (1, 21):
        myMatrix = MMatrix(i*0.05)
        myMatrix.generateMatrix()
        myMatrix.simulatePhenotypeInfinitesimally()
        myMatrix.generateModel(myMatrix.genotype_matrix, myMatrix.simulate_inf_phenotype, "infinitesimally")
    #myMatrix.simulatePhenotypeFinitesimally()
    #myMatrix.generateModel(myMatrix.genotype_matrix, myMatrix.simulate_uninf_phenotype, "finitesimally")
