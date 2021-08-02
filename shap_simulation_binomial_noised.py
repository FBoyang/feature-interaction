import numpy as np
import pandas as pd
import shap
import sklearn as sk
import sympy as sp
import matplotlib.pyplot as plt
import torch
import layers

class MMatrix:
    def __init__(self, g):
        #np.random.seed(12345)
        self.N = 1000 #np.random.randint(1000, 5001)
        self.M = 20 #np.random.randint(10, 101)

        self.inf_beta_array = np.array([])
        self.uninf_beta_array = np.array([])

        self.inf_epsilon_array = np.array([])
        self.uninf_epsilon_array = np.array([])

        self.sigma_g_squared = g #np.random.uniform(0, 0.5)
        self.sigma_e_squared = 1 - self.sigma_g_squared

        self.g = str(np.around(g, decimals=2))

        #self.dl_model = torch.load("model.pth")

    def generateMatrix(self):
        self.genotype_matrix = np.empty((self.N, self.M))

        for i in range(0, self.M):
            p = np.random.uniform(0, 1)
            for j in range(0, self.N):
                self.genotype_matrix[j][i] = np.random.binomial(2, p)

        #column-wise normalization of genotype matrix to have mean=0 and variance=1
        self.genotype_matrix = sk.preprocessing.scale(self.genotype_matrix)

    def simulatePhenotypeInfinitesimally(self):
        self.genotype_matrix = np.genfromtxt('outputs/genotype_0.1g.csv', delimiter=',')

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
    #     #post-noise simulated phenotype
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

        #Compute DeepSHAP Values
        # background = X[np.random.choice(X.shape[0], 100, replace=False)]
        # explainer = shap.DeepExplainer(self.dl_model, background)
        # shap_values = explainer.shap_values(X[0])

        # self.vals = np.abs(explainer.shap_values(X)).mean(0)

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

    def export(self, number):
        #code to export genotype, phenotype, and coefficient/shap value data
        #np.savetxt("outputs/genotype_" + self.g + "g.csv", self.genotype_matrix, delimiter=",")
        np.savetxt("outputs/phenotype_" + self.g + "g" + "_" + str(number) + ".csv", self.simulate_inf_phenotype_noised)
        #combined = np.transpose(np.vstack((self.inf_beta_array, self.coefficients, self.vals)))
        #np.savetxt("outputs/coefficients_and_shap_" + self.g + "g.csv", combined, delimiter=",", header="True Coefficients,sklearn Coefficients,Mean Abs. Shap Values")


        #code to generate dataframe with top N features, sorted by descending
        # n_features = 20
        # combined_abs_top_n = np.transpose(np.vstack((np.abs(self.inf_beta_array), np.abs(self.coefficients), self.vals)))[0:n_features]
        # comparison_df = pd.DataFrame(data=combined_abs_top_n, columns=["True Coefficient Absolute Value", "sklearn Coefficient Absolute Value", "SHAP Mean Absolute Value"],
        #                              index=["G"+str(k) for k in range(1, combined_abs_top_n.shape[0]+1)])
        # comparison_df = comparison_df.sort_values(by=['True Coefficient Absolute Value'], ascending=False)

        #code to calculate pearson and kendall coefficients between coefficients and shap values
        # pearson_matrix = comparison_df.corr(method='pearson')
        # pearson_corr = pearson_matrix["True Coefficient Absolute Value"]["SHAP Mean Absolute Value"]
        #
        # kendall_matrix = comparison_df.corr(method='kendall')
        # kendall_corr = kendall_matrix["True Coefficient Absolute Value"]["SHAP Mean Absolute Value"]
        #
        # correlation.append([pearson_corr, kendall_corr])

        #code to generate and save bar plot
        # ax = comparison_df.plot.bar(rot=0)
        # plt.savefig("plots/comparison_" + self.g + "g.png", dpi=1000)
        #plt.show()
        # plt.close()


if __name__ == "__main__":
    for i in range (1, 201):
        myMatrix = MMatrix(0.1)
        #myMatrix.generateMatrix()
        myMatrix.simulatePhenotypeInfinitesimally()
        myMatrix.generateModel(myMatrix.genotype_matrix, myMatrix.simulate_inf_phenotype_noised, "infinitesimally")
        myMatrix.export(i)