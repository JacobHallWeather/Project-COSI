import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LassoLars, OrthogonalMatchingPursuit
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

scoring = 'r2'
score_name = 'R-Sq'
    
def build_and_evaluate_pipeline(model, params, X, y, model_name, stats, dayType, use_rfecv):
    
    cv = KFold(n_splits=10, shuffle=True, random_state=1)  # Use 10-fold cross-validation
    
    if use_rfecv:
        # Use RFECV to select the best features
        selector = RFECV(model, cv=cv, scoring=scoring, n_jobs=30) 
        
        # Get the feature names before transforming X_train and X_test
        feature_names = X.columns
        
        # Fit and transform X
        X = selector.fit_transform(X, y)
        
        # Get the selected feature names
        selected_indices = np.where(selector.support_)[0]
        selected_features = feature_names[selected_indices]
        print(f"Selected features from RFECV: {', '.join(selected_features)} - {model_name}")
        
        # Convert X back to pandas dataframe
        X = pd.DataFrame(X, columns=selected_features)
        feature_names = X.columns
        
    else:
        # Print available variables with their corresponding index
        print("\nAvailable Variables:")
        for i, variable in enumerate(X.columns):
            print(f"{i + 1}. {variable}")
    
        while True:
            print("\nAvailable Variables:")
            for i, variable in enumerate(X.columns):
                print(f"{i + 1}. {variable}")
            variables_input = input("Enter the numbers of the variables you want to use (separated by commas): ").strip()
            try:
                selected_indices = [int(num) - 1 for num in variables_input.split(',')]
                if all(0 <= idx < len(X.columns) for idx in selected_indices):
                    break
                print("Invalid input, please try again.")
            except ValueError:
                print("Invalid input, please try again.")
            # Keep only the specified variables
            X = X.iloc[:, selected_indices]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('poly', PolynomialFeatures()), 
        ('regressor', model)
    ])

    # Perform GridSearchCV on the pipeline with the given hyperparameter grid
    grid = GridSearchCV(pipeline, params, cv=cv, iid=False, scoring=scoring, n_jobs=30)

    grid.fit(X, y)

    # Obtain the best estimator and evaluate its performance on the test set
    best_model = grid.best_estimator_
    scores = cross_val_score(best_model, X, y, scoring=scoring, cv=cv)
    score = np.mean(scores)


    # Print the final regression equation for the best estimator
    coefs = best_model.named_steps['regressor'].coef_[0] if len(best_model.named_steps['regressor'].coef_.shape) > 1 else best_model.named_steps['regressor'].coef_
    intercept = best_model.named_steps['regressor'].intercept_
    DEquation = f"y = {intercept:.2f}"
    equation = f"y = {intercept:.2f}"
    DsimpleEquation = f"y = {intercept:.2f}"
    print()
    for coef, feature in zip(coefs, selected_features):
        if feature in stats:
            mean = stats[feature]['mean']
            std = stats[feature]['std']
            if std != 0:
                DEquation += f" + {coef:.2f}*({feature}-{mean:.2f})/{std:.2f}"
                equation += f" + {coef:.2f}*{feature}"
                DsimpleEquation += f" + {coef/std:.2f}*({feature}-{mean:.2f})"
            else:
                DEquation += f" + {coef:.2f}*{feature}"
                equation += f" + {coef:.2f}*{feature}"
                DsimpleEquation += f" + {coef:.2f}*{feature}"
                
        else:
            DEquation += f" + {coef:.2f}*{feature}"
            equation += f" + {coef:.2f}*{feature}"
            DsimpleEquation += f" + {coef:.2f}*{feature}"

    return score, best_model, equation, DEquation, DsimpleEquation



def perform_regression(data, stats, dayType):
    X, y = data.drop("Attendance", axis=1), data["Attendance"]
    correlation_matrix = X.corr()
    
    print('Checking for multicollinearity')
    # Find correlations greater than 0.6
    high_correlations = correlation_matrix[((correlation_matrix > 0.5) & (correlation_matrix < 1.0)) | ((correlation_matrix < -0.5) & (correlation_matrix > -1.0))]
    high_correlations = high_correlations.stack().reset_index()
    high_correlations = high_correlations[high_correlations['level_0'] < high_correlations['level_1']]
    high_correlations.columns = ['Feature_1', 'Feature_2', 'Correlation']
    
    # Print the high correlations
    seen_combinations = set()
    for _, row in high_correlations.iterrows():
        combination = (row['Feature_1'], row['Feature_2'])
        if combination not in seen_combinations and combination[::-1] not in seen_combinations:
            seen_combinations.add(combination)
            print(f"Correlation between '{row['Feature_1']}' and '{row['Feature_2']}': {row['Correlation']}")



   
    models = {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'regressor__alpha': np.logspace(-4, 4, 50)}),
        'Lasso': (Lasso(), {'regressor__alpha': np.logspace(-5, 3, 100), 'regressor__max_iter': [2000000], 'regressor__tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}),
        'ElasticNet': (ElasticNet(), {'regressor__alpha': np.logspace(-4, 4, 50), 'regressor__max_iter': [400000], 'regressor__tol': [1e-4]}),
        'BayesianRidge': (BayesianRidge(), {}),
        'LARS': (LassoLars(), {'regressor__max_iter': [500], 'regressor__eps': [1e-14]}),
    }
    
    X_f = X.drop(columns=['Area', 'JulianDay'])
    
    while True:
        print("\nAvailable Variables:")
        for i, variable in enumerate(X_f.columns):
            print(f"{i + 1}. {variable}")
        variables_input = input("Enter the numbers of the variables you want to remove (separated by commas), or 'done' if you are finished: ").strip()
        if variables_input.lower() == 'done':
            break
        try:
            selected_indices = [int(num) - 1 for num in variables_input.split(',')]
            if all(0 <= idx < len(X_f.columns) for idx in selected_indices):
                X_f = X_f.drop(X_f.columns[selected_indices], axis=1)
                print("Variables removed.")
            else:
                print("Invalid input, please try again.")
        except ValueError:
            print("Invalid input, please try again.")                                                         

    results = Parallel(n_jobs=1)(delayed(build_and_evaluate_pipeline)(model, params, X_f, y, name, stats, dayType, RFECV)  # Pass name as an additional argument
                                   for name, (model, params) in models.items())

    feature_names = X_f.columns
    print()

    
    results_dict = {}

    for (name, (model, params)), (score, estimator, equation, DEquation, DsimpleEquation) in zip(models.items(), results):
        results_dict[name] = {score_name: score, 'Equation': equation, 'Display Equation': DEquation, 'Simple Equation': DsimpleEquation}
    
    results_dict = {k: {**v, 'Index': i+1} for i, (k, v) in enumerate(sorted(results_dict.items(), key=lambda item: item[1][score_name], reverse=True))}
    return results_dict, score_name


