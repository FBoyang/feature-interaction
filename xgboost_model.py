import xgboost
import numpy as np
import shap
from sklearn.model_selection import train_test_split

for i in range (1, 201):
    X = np.genfromtxt('outputs/0.1g/genotype_0.1g.csv', delimiter=',')
    y = np.genfromtxt('outputs/0.1g/phenotype_0.1g_' + str(i) + '.csv', delimiter=',')

    Xd = xgboost.DMatrix(data=X,label=y)

    model = xgboost.train({'eta':0.3, 'max_depth':4, 'base_score': 0, "lambda": 0}, Xd, 1)
    print("Model error =", np.linalg.norm(y-model.predict(Xd)))
    print(model.get_dump(with_stats=True)[0])

    pred = model.predict(Xd, output_margin=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xd)

    shap_interaction_values = explainer.shap_interaction_values(Xd)
    np.savetxt("xgboost_outputs/InteractionScore/XGBoost_" + str(i) + ".csv", shap_interaction_values[0], delimiter=",", fmt='%f')



