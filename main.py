import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, classification_report, precision_score, \
    recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


def read_dataset():
    df = pd.read_excel('Credit_cards_churners_GW2.xlsx')
    return df


def create_variables(df):
    # We transform the qualitative var into dummy variables (binary)
    df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda col: 1 if col == 'Attrited Customer' else 0)
    df['Gender'] = df['Gender'].apply(lambda col: 1 if col == 'M' else 0)
    df = pd.get_dummies(df, columns=['Education_Level'], prefix='edu')
    df = pd.get_dummies(df, columns=['Marital_Status'], prefix='mar')
    df = pd.get_dummies(df, columns=['Income_Category'], prefix='inc')
    df = pd.get_dummies(df, columns=['Card_Category'], prefix='card')
    df = df.dropna()
    df.set_index('CLIENTNUM', inplace=True)

    return df


def run_elastic_net():
    # We will use a subsample as a holdout set to evaluate the final model performance
    train_valid_sub, test_sub = train_test_split(train, test_size=0.15, random_state=40)
    t_test_sub = test_sub['Attrition_Flag']
    test_sub = test_sub.drop("Attrition_Flag", axis=1)

    # We Split again, to have a training and validation set.
    # We will use the validation set to tune the hyperparameters of the model
    train_sub, valid_sub = train_test_split(train_valid_sub, test_size=0.2, random_state=40)
    t_train_sub = train_sub['Attrition_Flag']
    train_sub = train_sub.drop("Attrition_Flag", axis=1)

    t_valid_sub = valid_sub['Attrition_Flag']
    valid_sub = valid_sub.drop("Attrition_Flag", axis=1)

    ################################################################
    ################################################################
    # SCALING
    # We now scale our quantitative variables so they're no overweighting
    scaler = StandardScaler()
    # Fit the scaler object to the training and validation data and transform the datas
    train_sub_scaled = scaler.fit_transform(train_sub[quant_cols])
    valid_sub_scaled = scaler.fit_transform(valid_sub[quant_cols])

    # Transform the testing data using the scaler object fitted on the training data
    test_sub_scaled = scaler.transform(test_sub[quant_cols])

    # Put the scaled data back into the original data frames
    train_sub[quant_cols] = train_sub_scaled
    valid_sub[quant_cols] = valid_sub_scaled
    test_sub[quant_cols] = test_sub_scaled

    ################################################################
    ################################################################
    # ELASTIC NET
    # We use cross validation on our elastic net to get the best hyperparameters
    model = ElasticNetCV(cv=10, random_state=40)
    model.fit(train_sub, t_train_sub)
    # Get the best hyperparameters
    best_alpha = model.alpha_
    best_l1_ratio = model.l1_ratio_

    # APPLY BEST HYPERPARAMS TO TRAINING SUBSAMPLE
    # Define the Elastic Net model with the best hyperparameters
    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
    model.fit(train_sub, t_train_sub)

    # Evaluate R² of training
    y_pred_training = model.predict(train_sub)
    score_train = model.score(train_sub, t_train_sub)

    # Evaluate the model on the validation sub
    y_pred = model.predict(valid_sub)
    score = model.score(valid_sub, t_valid_sub)

    # Evaluate the model on the testing sub
    y_pred_test = model.predict(test_sub)
    score_test = model.score(test_sub, t_test_sub)

    # Compute ROC curve and AUC score for y_pred (valid_sub)
    fpr, tpr, thresholds = roc_curve(t_valid_sub, y_pred)
    auc = roc_auc_score(t_valid_sub, y_pred)

    # Compute ROC curve and AUC score for y_pred_test (test_sub)
    fpr_test, tpr_test, thresholds_test = roc_curve(t_test_sub, y_pred_test)
    auc_test = roc_auc_score(t_test_sub, y_pred_test)

    # Plot ROC curves
    # plt.plot(fpr, tpr, label=f'Validation Data (AUC = {auc:.2f})')
    # plt.plot(fpr_test, tpr_test, label=f'Test Data (AUC = {auc_test:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend()
    # plt.show()

    # Print R²
    print(f'Training R²: {score_train:.2f}')
    print(f'Validation R²: {score:.2f}')
    print(f'Test R²: {score_test:.2f}')

    # Interpret the results
    coefficients_test = model.coef_
    # Create a DataFrame to hold the feature names and their corresponding coefficients
    coef_df_test = pd.DataFrame({'variable': train_sub.columns, 'coefficient': model.coef_})
    # Sort the coefficients & print
    coef_df_test = coef_df_test.reindex(coef_df_test['coefficient'].abs().sort_values(ascending=False).index)
    print(coef_df_test)

    # Sort the coefficients by absolute value
    sorted_coefficients = sorted(zip(coefficients_test, train_sub.columns), key=lambda x: abs(x[0]), reverse=True)

    # Extract the coefficients and feature names
    coefficients = [coef for coef, name in sorted_coefficients]
    names = [name for coef, name in sorted_coefficients]

    # # Plot the coefficients
    # plt.figure(figsize=(12, 8))
    # plt.bar(names, coefficients)
    # plt.xticks(rotation=90)
    # plt.xlabel('Features')
    # plt.ylabel('Coefficients')
    # plt.title('Elastic Net Coefficients')
    #
    # plt.subplots_adjust(bottom=0.4)
    # plt.show()

    return coef_df_test


def backward_selection_bic(X, y, nb_features):
    """
    Implements backward selection with BIC criterion.
    Returns a list of selected features.
    """
    # Initialize the set of selected features to include all features
    selected_features = list(X.columns)
    num_features = len(selected_features)

    # Create a logistic regression object
    logreg = LogisticRegression(penalty=None, solver='newton-cg', max_iter=1000)

    # Compute the initial BIC
    logreg.fit(X, y)
    y_hat = logreg.predict(X)
    n = len(y)
    ll = np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    k = num_features + 1  # plus one for the intercept term
    bic = -2 * ll + k * np.log(n)

    # Perform backward selection
    while num_features > nb_features:
        # Initialize the minimum BIC and the index of the feature to remove
        min_bic = float('inf')
        feature_to_remove = None

        # Iterate over the selected features and compute the BIC when removing each one
        for feature in selected_features:
            # Remove the feature from the design matrix
            X_temp = X.drop(columns=[feature])

            # Fit a logistic regression model and compute the BIC
            logreg.fit(X_temp, y)
            y_hat = logreg.predict(X_temp)
            y_hat = np.maximum(np.minimum(y_hat, 1 - 1e-15), 1e-15)
            ll = np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            k = num_features  # minus one for the removed feature
            bic_temp = -2 * ll + k * np.log(n)

            # Update the minimum BIC and the feature to remove
            if bic_temp < min_bic:
                min_bic = bic_temp
                feature_to_remove = feature

        # Remove the feature with the smallest BIC
        selected_features.remove(feature_to_remove)
        num_features -= 1

        # Update the BIC
        bic = min_bic

    # Return the final set of selected features
    print(selected_features)
    return selected_features


def knn_tuning(X, Y):
    # NB : Used to get best K value using cross-validation
    param_grid = {'n_neighbors': range(1, 30)}
    knn = KNeighborsClassifier()
    #
    # Use grid search to find the best k value
    # using scoring = 'accurracy' with the current subset of variables will return the same values
    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X, Y)
    print("Best k: ", grid_search.best_params_['n_neighbors'])
    return grid_search.best_params_['n_neighbors']


def knn_model(nb_neighbor, X_train, Y_train, X_valid, Y_valid):
    knn = KNeighborsClassifier(n_neighbors=nb_neighbor)
    knn.fit(X_train, Y_train)

    # Make predictions on the validation set
    y_predicted = knn.predict(X_valid)

    # Print metrics
    precision = precision_score(Y_valid, y_predicted)
    recall = recall_score(Y_valid, y_predicted)
    f1 = f1_score(Y_valid, y_predicted)
    accuracy = accuracy_score(Y_valid, y_predicted)
    roc_auc = roc_auc_score(Y_valid, y_predicted)
    mse = mean_squared_error(Y_valid, y_predicted)
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    print("Accuracy: {:.4f}".format(accuracy))
    print(f"ROC AUC score: {roc_auc:.4f}")
    print(f"MSE: {mse:.4f}")

    # cross-validation
    scores_reduced = cross_val_score(knn, X_valid, Y_valid, cv=5, scoring='roc_auc')
    # Print the mean and standard deviation of the scores
    print("Cross-validation scores on reduced variables:", scores_reduced)
    print("Mean score:", scores_reduced.mean())
    print("Standard deviation:", scores_reduced.std())

    # Calculate the ROC curves and AUC scores
    fpr, tpr, thresholds = roc_curve(Y_valid, knn.predict_proba(X_valid)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='KNN with reduced set of variables (AUC = %0.2f)' % roc_auc)
    # Add random guessing curve
    plt.plot([0, 1], [0, 1], 'k--')

    # Add axis labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc Curve for validation data')

    # Add legend and show plot
    plt.legend(loc="lower right")
    plt.show()


def knn_test_result(nb_neighbor, X_train, Y_train, variables_list):
    test_df = pd.read_excel('Credit_cards_churners_GW2.xlsx', sheet_name="Test sample")
    test_df = create_variables(test_df)

    # # Scale the features using the same scaler used for the training dataset
    test_scaled = scaler.transform(test_df[quant_cols])
    test_df[quant_cols] = test_scaled
    test_reduced = test_df[variables_list]

    # # Make predictions on the final dataset using the trained KNN model
    knn = KNeighborsClassifier(n_neighbors=nb_neighbor)
    knn.fit(X_train, Y_train)

    y_pred = knn.predict(test_reduced)
    # Compute classification report
    print(classification_report(test_df['Attrition_Flag'], y_pred))
    # Compute confusion matrix
    cm = confusion_matrix(test_df['Attrition_Flag'], y_pred)
    print("Confusion matrix:\n", cm)
    # Compute mean squared error
    mse = mean_squared_error(test_df['Attrition_Flag'], y_pred)
    print("Mean squared error:", mse)

    # Compute the ROC AUC score
    y_prob = knn.predict_proba(test_reduced)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_df['Attrition_Flag'], y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='KNN with reduced set of variables (AUC = %0.2f)' % roc_auc)

    # Add random guessing curve
    plt.plot([0, 1], [0, 1], 'k--')

    # Add axis labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on test sample')

    # Add legend and show plot
    plt.legend(loc="lower right")
    plt.show()


def random_forest_tuning(X, Y):
    # Hyperparameter tuning
    # CAUTION: THIS TAKES SEVERAL MINUTES !
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Define the scoring metric
    scoring = 'roc_auc'

    # Create a random forest model
    rf = RandomForestClassifier(random_state=40)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(rf, param_grid, scoring=scoring, cv=5)

    # Fit the model to the training data
    grid_search.fit(X, Y)

    # Print the best parameters and best score
    print("Best parameters: ", grid_search.best_params_)
    print("Best ROC AUC score (in CV over training sample): ", grid_search.best_score_)
    return grid_search.best_params_


def random_forest_model(estimator, depth, leaf, split, X_train, Y_train, X_valid, Y_valid):
    rf = RandomForestClassifier(n_estimators=estimator, max_depth=depth, min_samples_leaf=leaf, min_samples_split=split)
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_valid)

    precision = precision_score(Y_valid, y_pred)
    recall = recall_score(Y_valid, y_pred)
    f1 = f1_score(Y_valid, y_pred)
    accuracy = accuracy_score(Y_valid, y_pred)
    roc_auc = roc_auc_score(Y_valid, y_pred)
    mse = mean_squared_error(Y_valid, y_pred)
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    print("Accuracy: {:.4f}".format(accuracy))
    print(f"ROC AUC score: {roc_auc:.4f}")
    print(f"MSE: {mse:.4f}")

    scores = cross_val_score(rf, X_valid, Y_valid, cv=5, scoring='roc_auc')

    # Print the mean and standard deviation of the scores
    print("Cross-validation on reduced scores:", scores)
    print("Mean score:", scores.mean())
    print("Standard deviation:", scores.std())

    fpr, tpr, threshold = roc_curve(Y_valid, rf.predict_proba(X_valid)[:, 1])
    auc = roc_auc_score(Y_valid, rf.predict_proba(X_valid)[:, 1])

    plt.plot(fpr, tpr, label=f'Random Forest with reduced set of variables (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')

    plt.title('ROC Curve for validation sample')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def random_forest_test_result(estimator, depth, leaf, split, X_train, Y_train, variables_list):
    rf = RandomForestClassifier(n_estimators=estimator, max_depth=depth, min_samples_leaf=leaf, min_samples_split=split)
    rf.fit(X_train, Y_train)
    test_df = pd.read_excel('Credit_cards_churners_GW2.xlsx', sheet_name="Test sample")
    test_df = create_variables(test_df)
    test_reduced = test_df[variables_list]

    # Apply the trained model to the test data
    y_pred_proba = rf.predict_proba(test_reduced)[:, 1]

    y_pred = rf.predict(test_reduced)
    # Compute classification report
    print(classification_report(test_df['Attrition_Flag'], y_pred))
    # Compute confusion matrix
    cm = confusion_matrix(test_df['Attrition_Flag'], y_pred)
    print("Confusion matrix:\n", cm)
    # Compute mean squared error
    mse = mean_squared_error(test_df['Attrition_Flag'], y_pred)
    print("Mean squared error:", mse)

    # Calculate fpr, tpr and threshold values
    fpr, tpr, thresholds = roc_curve(test_df['Attrition_Flag'], y_pred_proba)
    auc_score = roc_auc_score(test_df['Attrition_Flag'], y_pred_proba)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='Randrom Forest with reduced set of variables AUC = {:.3f}'.format(auc_score))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on test sample')
    plt.legend()
    plt.show()


def cart_tuning(X, Y):
    param_grid = {'max_depth': range(1, 11),
                  'min_samples_split': range(2, 21)}

    # Create a CART model with the Gini criterion
    dt = DecisionTreeClassifier(criterion='gini', random_state=42)

    # Use 5-fold cross-validation to search for the optimal hyperparameters
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X, Y)
    print('Best hyperparameters for CART:', grid_search.best_params_)
    print('Best AUC score (obtained in CV over training subamples):', grid_search.best_score_)
    return grid_search.best_params_


def cart_model(depth, split, X_train, Y_train, X_valid, Y_valid):
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
    dt.fit(X_train, Y_train)

    # Make predictions on the validation set
    y_pred = dt.predict(X_valid)
    # Print metrics
    precision = precision_score(Y_valid, y_pred)
    recall = recall_score(Y_valid, y_pred)
    f1 = f1_score(Y_valid, y_pred)
    accuracy = accuracy_score(Y_valid, y_pred)
    roc_auc = roc_auc_score(Y_valid, y_pred)
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    print("Accuracy: {:.4f}".format(accuracy))
    print(f"ROC AUC score: {roc_auc:.4f}")

    # cross validation
    scores = cross_val_score(dt, X_valid, Y_valid, cv=5, scoring='roc_auc')
    # Print the mean and standard deviation of the scores
    print("Cross-validation scores:", scores)
    print("Mean score:", scores.mean())
    print("Standard deviation:", scores.std())

    fpr, tpr, thresholds = roc_curve(Y_valid, dt.predict_proba(X_valid)[:, 1],  pos_label=1)
    roc_auc = roc_auc_score(Y_valid, y_pred)

    # Plot the ROC curves
    plt.plot(fpr, tpr, label='CART with reduced set of selected Variables(AUC = %0.2f)' % roc_auc)
    # Add random guessing curve
    plt.plot([0, 1], [0, 1], 'k--')

    # Add axis labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc Curve for validation sample')

    # Add legend and show plot
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(25, 10))
    plot_tree(dt, filled=True, fontsize=4)
    plt.show()


def cart_test_result(depth, split, X_train, Y_train, variables_list):
    # CART model on reduced_sample
    dt_reduced = DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
    dt_reduced.fit(X_train, Y_train)

    test_df = pd.read_excel('Credit_cards_churners_GW2.xlsx', sheet_name="Test sample")
    test_df = create_variables(test_df)
    test_reduced = test_df[variables_list]
    # Compute the ROC AUC score
    y_prob = dt_reduced.predict_proba(test_reduced)[:, 1]

    fpr, tpr, thresholds = roc_curve(test_df['Attrition_Flag'], y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='CART with selected variables(AUC = %0.2f)' % roc_auc)

    # Add random guessing curve
    plt.plot([0, 1], [0, 1], 'k--')

    # Add axis labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on test sample')

    # Add legend and show plot
    plt.legend(loc="lower right")
    plt.show()


def logit_tuning(X, Y):
    # Define the hyperparameters to be tuned and their possible values
    param_grid = {'C': [0.1, 1, 5, 10, 20, 50, 100], 'penalty': ['l2']}

    # Create a logistic regression object
    lr = LogisticRegression()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(lr, param_grid, cv=5)

    # Fit the GridSearchCV object on the training data
    grid_search.fit(X, Y)

    # Print the best hyperparameters and validation score
    print('Best hyperparameters for Logistic regression:', grid_search.best_params_)
    print('Best AUC score (obtained in CV over training subamples):', grid_search.best_score_)
    return grid_search.best_params_


def logit_model(C, penalty, X_train, Y_train, X_valid, Y_valid):
    lr = LogisticRegression(C=C, penalty=penalty)
    lr.fit(X_train, Y_train)

    # Make predictions on the validation set
    y_predicted = lr.predict(X_valid)

    # Print metrics
    precision = precision_score(Y_valid, y_predicted)
    recall = recall_score(Y_valid, y_predicted)
    f1 = f1_score(Y_valid, y_predicted)
    accuracy = accuracy_score(Y_valid, y_predicted)
    roc_auc = roc_auc_score(Y_valid, y_predicted)
    mse = mean_squared_error(t_valid, y_predicted)
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    print("Accuracy: {:.4f}".format(accuracy))
    print(f"ROC AUC score: {roc_auc:.4f}")
    print(f"MSE: {mse:.4f}")

    # cross-validation
    scores_reduced = cross_val_score(lr, X_valid, Y_valid, cv=5, scoring='roc_auc')
    # Print the mean and standard deviation of the scores
    print("Cross-validation scores on reduced variables:", scores_reduced)
    print("Mean score:", scores_reduced.mean())
    print("Standard deviation:", scores_reduced.std())

    # Calculate the ROC curves and AUC scores
    fpr, tpr, thresholds = roc_curve(Y_valid, lr.predict_proba(X_valid)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curves
    plt.plot(fpr, tpr, label='Logit with reduced set of variables  (AUC = %0.2f)' % roc_auc)
    # Add random guessing curve
    plt.plot([0, 1], [0, 1], 'k--')

    # Add axis labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc Curve for validation sample')

    # Add legend and show plot
    plt.legend(loc="lower right")
    plt.show()


def logit_test_result(C, penalty, X_train, Y_train, variables_list):
    lr = LogisticRegression(C=C, penalty=penalty)
    lr.fit(X_train, Y_train)

    test_df = pd.read_excel('Credit_cards_churners_GW2.xlsx', sheet_name="Test sample")
    test_df = create_variables(test_df)
    test_reduced = test_df[variables_list]
    # Compute the ROC AUC score
    y_prob = lr.predict_proba(test_reduced)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_df['Attrition_Flag'], y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='CART with reduced set of variables (AUC = %0.2f)' % roc_auc)
    # Add random guessing curve
    plt.plot([0, 1], [0, 1], 'k--')
    # Add axis labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on test sample')

    # Add legend and show plot
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    train = read_dataset()
    # Part 1 : Training and validation sample
    # Q1 :
    # Select the columns containing quantitative variables, needed in elastic net resizing procedure
    quant_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                  'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                  'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                  'Avg_Utilization_Ratio']

    # transform qualitative variables into dummy variables
    train = create_variables(train)
    pd.options.display.max_columns = 1000

    # display summary information on the dataset
    summary = train.describe()
    corr_matrix = train.corr()
    print(summary.to_string())
    print(corr_matrix)

    # Q3
    #####################################################
    # ELASTIC NET VARIABLE SELECTION
    # we apply elastic net
    sorted_variables = run_elastic_net()
    # change the number X in sorted_variables.head(X) by the number of variables you want to selection from the elastic
    # net regression
    # all models will update accordingly to this line, which is crucial
    best_variables = sorted_variables.head(5)['variable']
    #####################################################

    # We split our data into train and validation sample
    # we will also compute dataframes with only the most "meaningful" variables as obtained through elastic net
    # and we will use  our models on both the full set of variables and the "reduced" subset of variables
    train_sample, valid_sample = train_test_split(train, test_size=0.2, random_state=42)
    t_train = train_sample['Attrition_Flag']
    t_valid = valid_sample['Attrition_Flag']

    train_sample = train_sample.drop("Attrition_Flag", axis=1)
    valid_sample = valid_sample.drop("Attrition_Flag", axis=1)

    train_sample_scaled = train_sample.copy()
    valid_sample_scaled = valid_sample.copy()
    ################################################################
    # SCALING
    # We now scale our quantitative variables so they're no overweighting
    scaler = StandardScaler()

    # Fit the scaler object to the training and validation data and transform the datas
    train_scaled = scaler.fit_transform(train_sample[quant_cols])
    valid_scaled = scaler.transform(valid_sample[quant_cols])
    # Put the scaled data back into the original data frames
    train_sample_scaled[quant_cols] = train_scaled
    valid_sample_scaled[quant_cols] = valid_scaled

    train_reduced = train_sample[best_variables]
    valid_reduced = valid_sample[best_variables]

    train_scaled_reduced = train_sample_scaled[best_variables]
    valid_scaled_reduced = valid_sample_scaled[best_variables]

    #####################################################
    # UNCOMMENT IF WILLING TO USE BACKWARD SELECTION
    # Backward selection of variables using BIC
    # backward_selection = backward_selection_bic(train_sample_scaled, t_train, 6)
    # train_s_bs = train_sample_scaled[backward_selection]
    # valid_s_bs = valid_sample_scaled[backward_selection]
    #
    # train_bs = train_sample[backward_selection]
    # valid_bs = valid_sample[backward_selection]
    ##################################################

    # # KNN model with variables from elastic net
    nb_neighbour = knn_tuning(train_scaled_reduced, t_train)
    knn_model(nb_neighbour, train_scaled_reduced, t_train, valid_scaled_reduced, t_valid)
    knn_test_result(nb_neighbour, train_scaled_reduced, t_train, best_variables)

    # # KNN model with variables frm backward selection
    # nb_neighbour = knn_tuning(train_s_bs, t_train)
    # knn_model(nb_neighbour, train_s_bs, t_train, valid_s_bs, t_valid)
    # knn_test_result(nb_neighbour, train_s_bs, t_train, backward_selection)

    # # CART model on with subset of variables coming from elastic net
    cart_params = cart_tuning(train_reduced, t_train)
    cart_model(cart_params['max_depth'], cart_params['min_samples_split'], train_reduced, t_train, valid_reduced, t_valid)
    cart_test_result(cart_params['max_depth'], cart_params['min_samples_split'], train_reduced, t_train, best_variables)

    # # CART model on with subset of variables coming from backward selection
    # train_bs = train_sample_scaled[backward_selection]
    # valid_bs = valid_sample_scaled[backward_selection]
    # cart_params = cart_tuning(train_bs, t_train)
    # cart_model(cart_params['max_depth'], cart_params['min_samples_split'], train_bs, t_train, valid_bs, t_valid)
    # cart_test_result(cart_params['max_depth'], cart_params['min_samples_split'], train_bs, t_train, backward_selection)

    # # RANDOM FOREST model on with subset of variables coming from Elastic Net
    # # rf_params = random_forest_tuning()    # The tuning is very long, we do not run it everytime
    # # we assign the values ourselves as the tuning function takes between 5 and 10 min
    rf_params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
    # print(rf_params)
    random_forest_model(rf_params['n_estimators'], rf_params['max_depth'], rf_params['min_samples_leaf'],
                        rf_params['min_samples_split'], train_reduced, t_train, valid_reduced, t_valid)
    random_forest_test_result(rf_params['n_estimators'], rf_params['max_depth'], rf_params['min_samples_leaf'],
                            rf_params['min_samples_split'], train_reduced, t_train, best_variables)

    # # # RANDOM FOREST model on with subset of variables coming from backward selection
    # random_forest_model(rf_params['n_estimators'], rf_params['max_depth'], rf_params['min_samples_leaf'],
    #                     rf_params['min_samples_split'], train_bs, t_train, valid_bs, t_valid)
    # random_forest_test_result(rf_params['n_estimators'], rf_params['max_depth'], rf_params['min_samples_leaf'],
    #                          rf_params['min_samples_split'], train_bs, t_train, backward_selection)

    # # Logistic model with variables from elastic net
    log_params = logit_tuning(train_scaled_reduced, t_train)
    logit_model(log_params['C'], log_params['penalty'], train_scaled_reduced, t_train, valid_scaled_reduced, t_valid)
    logit_test_result(log_params['C'], log_params['penalty'], train_scaled_reduced, t_train, best_variables)

    # # Logistic model with variables from backward selection net
    # log_params = logit_tuning(train_scaled_reduced, t_train)
    # logit_model(log_params['C'], log_params['penalty'], train_s_bs, t_train, valid_s_bs, t_valid)
    # logit_test_result(log_params['C'], log_params['penalty'], train_s_bs, t_train, backward_selection)
