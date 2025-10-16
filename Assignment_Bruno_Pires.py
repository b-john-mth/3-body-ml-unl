def main():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge, RidgeCV, LassoCV
    from sklearn.linear_model import MultiTaskLassoCV
    from sklearn.compose import ColumnTransformer
    import seaborn as sns
    import time
    import warnings
    warnings.filterwarnings("ignore")

    #TASK 1: Setting the Baseline   --------------------------------------------------------------------------------------------
    # TASK 1.1: Data Preparation and Validation Pipeline
    # Load the data
    train_data = pd.read_csv('X_train.csv')
    test_data = pd.read_csv('X_test.csv')

    # Exploration of the training data
    # Visualization of the trajectories 
    idx = np.hstack((0,train_data[train_data.t == 10].index.values +1))
    k = np.random.randint(idx.shape[0])
    print(k)
    pltidx = range(idx[k],257+idx[k])
    pltsquare = idx[k]
    plt.plot(train_data.x_1[pltidx], train_data.y_1[pltidx])
    plt.plot(train_data.x_2[pltidx], train_data.y_2[pltidx])
    plt.plot(train_data.x_3[pltidx], train_data.y_3[pltidx]) 
    plt.plot(train_data.x_1[pltsquare], train_data.y_1[pltsquare],'s')
    plt.plot(train_data.x_2[pltsquare], train_data.y_2[pltsquare],'s')
    plt.plot(train_data.x_3[pltsquare], train_data.y_3[pltsquare],'s')   
    plt.title('Training Data - Trajectory visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    plt.savefig('TrainData_Trajectory4.png')
    # remove unwanted data
    train_data = train_data.drop(columns=['Id', 'v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'])
    # Extract features and targets from the training data and rename columns to match test data
    test_data.rename(columns={'x0_1':'x_1', 'y0_1':'y_1', 'x0_2':'x_2', 'y0_2':'y_2', 'x0_3':'x_3', 'y0_3':'y_3'}, inplace=True)
    input_features = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    target_variables = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    X_test = test_data[input_features]
    #Auxiliary functions to prepare the dataset and split into training and validation sets based on trajectories
    def build_train_dataset(df, traj_len=257):
        df = df.reset_index(drop=True)
        n_rows = len(df)
        n_traj = n_rows // traj_len

        X_list, y_list = [], []

        for i in range(n_traj):
            start = i * traj_len
            end = start + traj_len
            traj = df.iloc[start:end]

            if len(traj) < traj_len:
                continue

            # All columns except 't' from row 0
            X0 = traj.iloc[0].drop('t')

            # Build X: repeat X0 for traj_len - 1 rows
            X_traj = pd.DataFrame([X0.values] * (traj_len - 1), columns=X0.index)

            # Assign t from row 0 to row 255
            X_traj['t'] = traj['t'].iloc[:traj_len - 1].values

            # Build y: rows 1 to traj_len - 1, drop t
            y_traj = traj.iloc[1:].drop(columns=['t']).reset_index(drop=True)

            X_list.append(X_traj)
            y_list.append(y_traj)

        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)

        return X, y

    def split_Xy_by_trajectory(X, y, traj_len=256, test_size=0.2, random_state=42):
        # Convert to DataFrame if needed
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        y_df = pd.DataFrame(y) if isinstance(y, np.ndarray) else y.copy()
        
        n_rows = len(X_df)
        n_traj = n_rows // traj_len
        
        # Assign trajectory ID by row index
        X_df['trajectory_id'] = np.repeat(np.arange(n_traj), traj_len)
        
        # Split trajectory IDs
        traj_ids = X_df['trajectory_id'].unique()
        train_ids, val_ids = train_test_split(traj_ids, test_size=test_size, random_state=random_state)
        
        # Filter rows by trajectory IDs
        train_mask = X_df['trajectory_id'].isin(train_ids)
        val_mask   = X_df['trajectory_id'].isin(val_ids)
        
        X_train_df = X_df.loc[train_mask].sort_values(['trajectory_id', 't'])
        X_val_df   = X_df.loc[val_mask].sort_values(['trajectory_id', 't'])
        y_train_df = y_df.loc[X_train_df.index]
        y_val_df   = y_df.loc[X_val_df.index]
        
        # Drop helper column
        X_train = X_train_df.drop(columns='trajectory_id').to_numpy()
        X_val   = X_val_df.drop(columns='trajectory_id').to_numpy()
        y_train = y_train_df.to_numpy()
        y_val   = y_val_df.to_numpy()
        
        return X_train, X_val, y_train, y_val
    
    def sample_trajectories(train_data, frac=0.01, replace=True, random_state=1, drop_trajectory_id=True):
        train_data = train_data.copy()
        
        # Assign trajectory IDs based on blocks of 257 rows
        trajectory_length = 257
        train_data['trajectory_id'] = train_data.index // trajectory_length
        
        # Sample trajectory IDs
        unique_trajectories = train_data['trajectory_id'].unique()
        sampled_traj_ids = pd.Series(unique_trajectories).sample(
            frac=frac,
            replace=replace,
            random_state=random_state
        )
        
        # Filter the dataset
        sampled_data = train_data[train_data['trajectory_id'].isin(sampled_traj_ids)].copy()
        sampled_data = sampled_data.sort_values(by=['trajectory_id', 't']).reset_index(drop=True)
        
        # Optionally drop trajectory_id
        if drop_trajectory_id:
            sampled_data = sampled_data.drop(columns='trajectory_id')
        
        return sampled_data

    X, y = build_train_dataset(train_data[input_features])
    X_train, X_val, y_train, y_val = split_Xy_by_trajectory(X, y, test_size=0.2, random_state=42)

    # TASK 1.2: Learn the baseline model
    # Create a pipeline with StandardScaler and LinearRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('regressor', LinearRegression())  # Linear Regression model
    ])
    # Train the model
    pipeline.fit(X_train, y_train)
    # Make predictions on the validation and test sets
    y_pred_train = pipeline.predict(X_train)
    y_pred_val = pipeline.predict(X_val)
    y_pred_test = pipeline.predict(X_test)
    # Evaluate the model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_val, y_pred_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    def plot_y_yhat(y_val, y_pred, plot_title):
        plot_title = plot_title
        labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
        MAX = 500
        # Ensure y_val and y_pred are at least 2D arrays
        y_val = np.atleast_2d(y_val)
        y_pred = np.atleast_2d(y_pred)
        if y_val.shape[0] > MAX:
            idx = np.random.choice(y_val.shape[0], MAX, replace=False)
        else:
            idx = np.arange(y_val.shape[0])
        plt.figure(figsize=(10, 10))
        for i in range(min(6, y_val.shape[1])):
            x0 = np.min(y_val[idx, i])
            x1 = np.max(y_val[idx, i])
            plt.subplot(3, 2, i + 1)
            plt.scatter(y_val[idx, i], y_pred[idx, i])
            plt.xlabel('True ' + labels[i])
            plt.ylabel('Predicted ' + labels[i])
            plt.plot([x0, x1], [x0, x1], color='red')
            plt.title(f'True vs Predicted for {plot_title} - {labels[i]}')
            plt.axis('square')
        plt.savefig(plot_title+'.pdf')
        plt.show()
    
    # Print the results
    print(f"Baseline Training MSE: {train_mse}", f"Training RMSE: {train_rmse}")
    print(f"Baseline Validation MSE: {val_mse}", f"Validation RMSE: {val_rmse}")
    plot_y_yhat(y_val, y_pred_val, plot_title='Baseline model')
    np.savetxt('baseline-model.csv', y_pred_test, delimiter=',')

    # TASK 2: Nonlinear models on the data — the Polynomial Regression model ---------------------------------------------------------
    # TASK 2.1: Development
    # Sample the data for 1% to run faster
    train_data_sampled = sample_trajectories(train_data, frac=0.01, replace=True, random_state=1, drop_trajectory_id=True)
    X_sample, y_sample = build_train_dataset(train_data_sampled[input_features])
    X_train_sampled, X_val_sampled, y_train_sampled, y_val_sampled = split_Xy_by_trajectory(X_sample, y_sample, test_size=0.2, random_state=42)
    # Function to validate polynomial regression with different degrees and optional feature limitation
    def validate_poly_regression(X_train, y_train, X_val, y_val, X_test, regressor=None, degrees=range(1, 15), max_features=None):
        best_val_rmse = float('inf')
        best_pipeline = None
        y_pred_test = None
        y_pred_val = None
        for degree in degrees:
            print(f"Testing degree {degree}...")
            pipeline = Pipeline([
                ('poly_features', PolynomialFeatures(degree=degree)),
                ('regressor', regressor if regressor else LinearRegression())
            ])
            # Fit the model on the training set
            pipeline.fit(X_train, y_train)
            # Make predictions on the training and validation set
            y_train_pred = pipeline.predict(X_train)
            y_val_pred = pipeline.predict(X_val)
            # Calculate RMSE for the validation set
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            # Store the best model if this degree yields a lower validation RMSE
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_train_rmse = train_rmse
                best_pipeline = pipeline
                y_train_pred = best_pipeline.predict(X_train)
                y_pred_test = best_pipeline.predict(X_test)
                y_pred_val = best_pipeline.predict(X_val)

            print(f"Degree {degree}: Validation RMSE = {val_rmse:.4f}")
        return best_pipeline, best_train_rmse, best_val_rmse, y_pred_test, y_pred_val
        
    best_degrees = []
    for i in range(10):
        best_pipeline, best_train_rmse, best_val_rmse, y_pred_test_poly, y_val_pred_sampled = validate_poly_regression(
            X_train_sampled,
            y_train_sampled,
            X_val_sampled,
            y_val_sampled,
            X_test,
            regressor=LinearRegression(),
            degrees=range(1,15))
        # Extracting the degree of the polynomial used in the best model
        poly_features = best_pipeline.named_steps['poly_features']
        best_degree = poly_features.degree
        best_degrees.append(best_degree)
        i+=1
    np.savetxt('polynomial_submission.csv', y_pred_test_poly, delimiter=',')

    # Plotting the distribution of selected polynomial degrees
    plt.figure(figsize=(10, 6))
    plt.hist(best_degrees, bins=np.arange(1, 16)-0.5, edgecolor='black', align='mid')
    plt.xticks(np.arange(1, 15))
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Frequency')
    plt.title('Distribution of Selected Polynomial Degrees over 10 Runs')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    # Output the best degree from the analysis
    unique_degrees, counts = np.unique(best_degrees, return_counts=True)
    best_degree_count = dict(zip(unique_degrees, counts))
    # Selecting the best degree based on frequency
    best_degree = max(best_degree_count, key=best_degree_count.get)
    #plot_y_yhat(y_val_sampled, y_val_pred_poly, plot_title='Test and validation output polinomial')
    print("Best Polynomial Degree based on Distribution:", best_degree)
    print("Best Train RMSE based on Distribution:", best_train_rmse)
    print("Best Validation RMSE based on Distribution:", best_val_rmse)
    plot_y_yhat(y_val_sampled, y_val_pred_sampled, plot_title='Polinomial model')

    # TASK 2.2 Evaluation
    # Call the validation function with RidgeCV
    ridge_regressor = RidgeCV(alphas=np.logspace(-6, 6, 13))  # RidgeCV with a range of alpha values
    best_pipeline_ridge, best_t_rmse_ridge, best_v_rmse_ridge, y_pred_test_ridge, y_val_pred_ridge = validate_poly_regression(
        X_train_sampled,
        y_train_sampled,
        X_val_sampled,
        y_val_sampled,
        X_test,
        regressor=ridge_regressor,
        degrees=range(1, 3))
    # Output the results for RidgeCV
    np.savetxt('ridge_polynomial_submission.csv', y_pred_test_ridge, delimiter=',')
    plot_y_yhat(y_val_sampled, y_val_pred_ridge, plot_title='Polinomial-Rigde')
    print("\nRidge Regression Results:")
    print("Best Train RMSE (Ridge): ", best_t_rmse_ridge)
    print("Best Validation RMSE (Ridge): ", best_v_rmse_ridge)
    print("Best Polynomial Degree (Ridge):", best_pipeline_ridge.named_steps['poly_features'].degree)
    # Call the validation function with LassoCV 
    lasso_regressor = MultiTaskLassoCV(alphas=np.logspace(-6, 6, 13))  # MultiTaskLassoCV with a range of alpha values
    best_pipeline_lasso, best_t_rmse_lasso, best_v_rmse_lasso, y_pred_test_lasso,  y_val_pred_lasso = validate_poly_regression(
        X_train_sampled,
        y_train_sampled,
        X_val_sampled,
        y_val_sampled,
        X_test,
        regressor=lasso_regressor,
        degrees=range(1, 3))
    # Output the results for LassoCV
    np.savetxt('lasso_polynomial_submission.csv', y_pred_test_lasso, delimiter=',')
    plot_y_yhat(y_val_sampled, y_val_pred_lasso, plot_title='Polinomial-Lasso')
    print("\nLasso Regression Results:")
    print("Best Train RMSE (Lasso): ", best_t_rmse_lasso)
    print("Best Validation RMSE (Lasso): ", best_v_rmse_lasso)
    print("Best Polynomial Degree (Lasso):", best_pipeline_lasso.named_steps['poly_features'].degree)

    # TASK 3: Feature Engineering
    # Task 3.1: Explore linear relationships and correlation between variables
    sns.pairplot(train_data.sample(200), kind="hist")
    corr = train_data.corr()
    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature and Target Correlation Heatmap")
    plt.show()
    
    # Get the upper triangle of the correlation matrix (to avoid duplicates)
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Stack the upper triangle matrix into a series and reset index
    correlation_series = upper_triangle.stack().reset_index()

    # Rename the columns for clarity
    correlation_series.columns = ['Variable1', 'Variable2', 'Correlation']

    # Sort by the absolute value of the correlation
    sorted_correlations = correlation_series.reindex(correlation_series['Correlation'].abs().sort_values(ascending=False).index)
    df_sorted_corr = pd.DataFrame(sorted_correlations)
    #df_sorted_corr.to_excel('correlation_results.xlsx', index=False)
    print(df_sorted_corr.head(10))
    
    # Task 3.2: Variable Reduction
    # Based on the correlation analysis, we will drop the third body's position variables (x_3, y_3, v_x_3, v_y_3)
    input_features_reduced = ['t', 'x_1', 'y_1', 'x_2', 'y_2']
    target_variables_reduced = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    X_sample_reduced = X_sample[input_features_reduced]
    y_sample_reduced = y_sample[target_variables_reduced]
    X_train_reduced, X_val_reduced, y_train_reduced, y_val_reduced = split_Xy_by_trajectory(X_sample_reduced, y_sample_reduced, test_size=0.2, random_state=42)
    X_test_reduced = X_test[input_features_reduced]
    best_pipeline_ridge_r, best_t_rmse_ridge, best_v_rmse_ridge, y_pred_test_reduced, y_val_pred_reduced = validate_poly_regression(
            X_train_reduced,
            y_train_reduced,
            X_val_reduced,
            y_val_reduced,
            X_test_reduced,
            regressor=ridge_regressor,
            degrees=range(1, 3))
    np.savetxt('reduced_polynomial_submission.csv', y_pred_test_reduced, delimiter=',')
    plot_y_yhat(y_val_reduced, y_val_pred_reduced, plot_title='Polinomial reduced')
    print("\nRidge Regression Results for reduced polynomial:")
    print("Best Train RMSE (Ridge) for reduced polynomial: ", best_t_rmse_ridge)
    print("Best Validation RMSE (Ridge) for reduced polynomial: ", best_v_rmse_ridge)
    print("Best Polynomial Degree (Ridge) for reduced polynomial:", best_pipeline_ridge_r.named_steps['poly_features'].degree)

    # Task 3.3: Add new features based on distance
    def add_rel_time_features(X):

        # Convert to DataFrame
        df = pd.DataFrame(X, columns=['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

        # === Temporal features ===
        df['sin_t'] = np.sin(2 * np.pi * df['t'] / df['t'].max())
        df['cos_t'] = np.cos(2 * np.pi * df['t'] / df['t'].max())

        # === Relative distances and sin/cos of angles ===
        pairs = [(1,2), (1,3), (2,3)]
        for i, j in pairs:
            dx = df[f'x_{i}'] - df[f'x_{j}']
            dy = df[f'y_{i}'] - df[f'y_{j}']

            # Distance
            df[f'd_{i}{j}'] = np.sqrt(dx**2 + dy**2)

            # Angle encoded as sin and cos
            df[f'theta_{i}{j}_sin'] = np.sin(np.arctan2(dy, dx))
            df[f'theta_{i}{j}_cos'] = np.cos(np.arctan2(dy, dx))

        return df

    train_feat_eng = add_rel_time_features(train_data[input_features])
    train_data_sampled = sample_trajectories(train_feat_eng, frac=0.01, replace=True, random_state=1, drop_trajectory_id=True)
    X_sample, y_sample = build_train_dataset(train_data_sampled)
    X_train_new_var, X_val_new_var, y_train_new_var, y_val_new_var = split_Xy_by_trajectory(X_sample, y_sample, test_size=0.2, random_state=42)
    X_test_new_var = add_rel_time_features(X_test[input_features])
    best_pipeline_ridge_a, best_t_rmse_ridge_a, best_v_rmse_ridge_a, y_pred_test_new, y_val_pred_new = validate_poly_regression(
        X_train_new_var,
        y_train_new_var,
        X_val_new_var,
        y_val_new_var,
        X_test_new_var,
        regressor=ridge_regressor,
        degrees=range(1, 3))
    #y_pred_test_mass = best_pipeline_ridge_a.predict(X_test_with_mass)
    np.savetxt('augmented_polynomial_submission.csv', y_pred_test_new, delimiter=',')
    plot_y_yhat(y_val_new_var, y_val_pred_new, plot_title='Polinomial augmented')
    print("\Polinomial Regression Results for augmented features:")
    print("Best Train RMSE for reduced polynomial: ", best_t_rmse_ridge_a)
    print("Best Validation RMSE for reduced polynomial: ", best_v_rmse_ridge_a)
    print("Best Polynomial Degree for augmented polynomial:", best_pipeline_ridge_a.named_steps['poly_features'].degree)

    # TASK 4: K-Nearest Neighbors Regression

    # KNN Validation Function --------------------------------------------------------------------------------
    def validate_knn_regression(X_train, y_train, X_val, y_val, X_test, k=range(1, 15)):
        best_rmse = float('inf')
        results = []  # To store results for each k
        for k_value in k:
            # Initialize the KNN regressor
            knn_model = KNeighborsRegressor(n_neighbors=k_value, algorithm='ball_tree', n_jobs=-1)

            # Measure training time
            start_time = time.time()
            knn_model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Measure inference time
            start_time = time.time()
            y_train_pred = knn_model.predict(X_train)
            y_val_pred = knn_model.predict(X_val)
            inference_time_val = time.time() - start_time

            # Calculate evaluation metrics for train and validation set
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_rmse = np.sqrt(train_mse)
            val_rmse = np.sqrt(val_mse)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            # Measure inference time on the valdiation set
            start_time = time.time()
            y_test_pred = knn_model.predict(X_test)
            y_val_pred = knn_model.predict(X_val)
            inference_time_test = time.time() - start_time
            
            # Store results
            results.append({
                'k': k_value,
                'Train MSE': val_mse,
                'Val MSE': val_mse,
                'Train RMSE': train_rmse,
                'Val RMSE': val_rmse,
                'Train R²': train_r2,
                'Val R²': val_r2,
                'Training Time (s)': training_time,
                'Inference Time Val (s)': inference_time_val,
                'Inference Time Test (s)': inference_time_test
            })
            # Update the submission prediction output
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                y_test_pred = knn_model.predict(X_test)

        # Create a DataFrame to store results
        results_df = pd.DataFrame(results)
        return results_df, y_test_pred

    knn_results, y_test_pred_knn, y_val_pred_knn = validate_knn_regression(X_train, y_train, X_val, y_val, X_test, k=range(1, 15))
    np.savetxt('knn_submission.csv', y_test_pred_knn, delimiter=',')
    plot_y_yhat(y_val, y_val_pred_knn, plot_title='Test and validation output KNN')
    print("\nResults of kNN Regression Validation:")
    print(knn_results)

main()