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
            p = np.random.uniform(0, 1)
            for j in range(0, self.N):
                self.genotype_matrix[j][i] = np.random.binomial(2, p)

        #column-wise normalization of genotype matrix to have mean=0 and variance=1
        self.genotype_matrix = sk.preprocessing.scale(self.genotype_matrix)

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

        self.coefficients = model.coef_

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


        self.vals = np.abs(explainer.shap_values(X)).mean(0)

        # shap.plots.beeswarm(shap_values, max_display=20, show=False)
        # plt.savefig("plots/" + modelType + "_summary_" + ".png")
        # plt.close()

        #2.2. Make a standard partial dependence plot at a specific index
        sample_ind = 18
        # fig, ax = shap.plots.partial_depenedence_plot("G0", model.predict, X,
        #        model_expected_value = True, feature_expected_value = True,
        #        show=False,ice=False,shap_values=shap_values[sample_ind:
        #        sample_ind+1,:],shap_value_features=X.iloc[sample_ind:
        #        sample_ind+1,:] )
        # fig.show()

        #3. SHAP plot waterfall
        # shap.plots.waterfall(shap_values[sample_ind], max_display=20, show=False)
        # plt.savefig("plots/" + modelType + "_waterfall_" + "18" + ".png")
        # plt.close()

        #Absolute value bar plot
        # shap.plots.bar(shap_values, max_display=20)
        # plt.savefig("plots/" + modelType + "_bar_" + self.g + "g" + ".png")
        # plt.close()

    def export(self):
        np.savetxt("outputs/genotype_" + self.g + "g.csv", self.genotype_matrix, delimiter=",")
        np.savetxt("outputs/phenotype_" + self.g + "g.csv", self.simulate_inf_phenotype_noised)

        combined = np.transpose(np.vstack((self.inf_beta_array, self.coefficients, self.vals)))
        np.savetxt("outputs/coefficients_and_shap_" + self.g + "g.csv", combined, delimiter=",", header="True Coefficients,sklearn Coefficients,Mean Abs. Shap Values")


        #code to generate comparison bar plots
        combined_abs = np.transpose(np.vstack((np.abs(self.inf_beta_array), np.abs(self.coefficients), self.vals)))
        headers = ["True Coefficient Absolute Value", "sklearn Coefficient Absolute Value", "SHAP Mean Absolute Value"]

        x = np.arange(combined_abs.shape[0])
        dx = (np.arange(combined_abs.shape[1]) - combined_abs.shape[1] / 2.) / (combined_abs.shape[1] + 2.)
        d = 1. / (combined_abs.shape[1] + 2.)

        fig, ax = plt.subplots()
        for i in range(combined_abs.shape[1]):
            ax.bar(x + dx[i], combined_abs[:, i], width=d, label=headers[i])

        plt.legend(framealpha=1)
        plt.savefig("plots/comparison_" + self.g + "g.png", dpi=1000)
        plt.close()


if __name__ == "__main__":
    for i in range(0, 6):
        myMatrix = MMatrix(i/10)
        myMatrix.generateMatrix()
        myMatrix.simulatePhenotypeInfinitesimally()
        myMatrix.generateModel(myMatrix.genotype_matrix, myMatrix.simulate_inf_phenotype, "infinitesimally")
        myMatrix.export()
