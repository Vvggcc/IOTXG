import pandas as pd
import numpy as np # non numerical data was observed
import os

# Define the dataset path
dataset_path = 'manufacturing_6G_dataset.csv'

def load_dataset(path):
    """
    Reads the manufacturing_6G_dataset into a multi-array (NumPy) and returns cleaning stats.
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, 0, 0
    
    # Load the dataset using pandas
    df = pd.read_csv(path, header=None)
    
    # Identify empty rows before dropping duplicates
    empty_rows_count = df.isnull().any(axis=1).sum()
    df = df.dropna()
    
    # Track duplicates after dropping NaNs
    post_nan_count = len(df)
    df = df.drop_duplicates()
    duplicates_count = post_nan_count - len(df)
    
    # Convert to NumPy multi-array
    multi_array = df.to_numpy()
    
    return multi_array, duplicates_count, empty_rows_count

if __name__ == "__main__":
    data, duplicates, empty_rows = load_dataset(dataset_path)
    
    if data is not None:
        print(f"Dataset loaded successfully!")
        print(f"Shape of multi-array: {data.shape}")
        
        # --- DATASET SUMMARY ---
        print("\n--- DATASET SUMMARY ---")
        print(f"1. Duplicates found and removed: {duplicates}")
        print(f"2. Empty rows (NaN) found and removed: {empty_rows}")
        
        # Target Variable Analysis (Efficiency_Status)
        # Efficiency_Status is at index 12 in each row. data[0] is the header.
        if data.shape[0] > 1:
            eff_column = data[1:, 12] # Skip header row
            unique_vals, counts = np.unique(eff_column, return_counts=True)
            eff_counts = dict(zip(unique_vals, counts))
            
            print(f"3. Target Variable Analysis (Efficiency_Status):")
            print(f"   - Low Efficiency:    {eff_counts.get('Low', 0)}")
            print(f"   - Medium Efficiency: {eff_counts.get('Medium', 0)}")
            print(f"   - High Efficiency:   {eff_counts.get('High', 0)}")
            
            # --- DATASET SHUFFLING AND STRATIFICATION ---
            
            # Convert multi-array back to DataFrame for stratification (excluding header row)
            df_cols = data[0]
            df_records = data[1:]
            full_df = pd.DataFrame(df_records, columns=df_cols)
            
            # --- ONE-HOT ENCODING FOR OPERATION_MODE ---
            # Create two new columns: Active_mode and Idle_mode
            # Value is 1 if matching the mode, 0 otherwise
            full_df['Active_mode'] = (full_df['Operation_Mode'] == 'Active').astype(int)
            full_df['Idle_mode'] = (full_df['Operation_Mode'] == 'Idle').astype(int)
            
            # Drop the original Operation_Mode column since it's been encoded
            full_df = full_df.drop('Operation_Mode', axis=1)
            
            # --- ORDINAL ENCODING FOR EFFICIENCY_STATUS ---
            # Efficiency status is ordinal, substituting values as requested
            efficiency_mapping = {'Low': 0, 'Medium': 0.5, 'High': 1}
            full_df['Efficiency_Status'] = full_df['Efficiency_Status'].map(efficiency_mapping)
            

            # 1. Randomize the dataset to ensure no ordering bias
            full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # 2. Stratify the data set (preserving Efficiency_Status ratios)
            target_col = 'Efficiency_Status'
            
            # Training set (70% of the total dataset)
            train_df = full_df.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(frac=0.7, random_state=42)
            )
            
            # Identify the remaining 30% for Testing and Validation
            remaining_df = full_df.drop(train_df.index)
            
            # Split the remaining 30% equally into Validation (15% total) and Testing (15% total)
            val_df = remaining_df.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(frac=0.5, random_state=42)
            )
            
            # Testing set is the final remaining portion
            test_df = remaining_df.drop(val_df.index)
            
            # 3. Save the stratified datasets into separate files for later use
            train_df.to_csv('train_set.csv', index=False)
            test_df.to_csv('test_set.csv', index=False)
            val_df.to_csv('val_set.csv', index=False)
            
            print(f"\n--- STRATIFICATION COMPLETE ---")
            print(f"Training set:   {len(train_df)} rows (saved to train_set.csv)")
            print(f"Testing set:    {len(test_df)} rows (saved to test_set.csv)")
            print(f"Validation set: {len(val_df)} rows (saved to val_set.csv)")
        else:
            print("No data rows available for analysis.")

    # --- MULTICOLLINEARITY ANALYSIS (TRAINING SET) ---
    print("\n--- MULTICOLLINEARITY ANALYSIS (TRAINING SET) ---")
    if os.path.exists('train_set.csv'):
        # Load the training set
        train_analysis_df = pd.read_csv('train_set.csv')
        
        # Drop the Timestamp and Machine_ID columns as requested
        cols_to_drop = ['Timestamp', 'Machine_ID']
        train_analysis_df = train_analysis_df.drop(columns=[col for col in cols_to_drop if col in train_analysis_df.columns])
        
        # Calculate the correlation matrix (Multicollinearity)
        corr_matrix = train_analysis_df.corr()
        
        # Save the multicollinearity matrix to a CSV file
        corr_matrix.to_csv('Multi-Collinearity.csv')
        print("Multicollinearity matrix saved to 'Multi-Collinearity.csv'")
        
        # Configure Pandas to print nicely as a table in the terminal
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', lambda x: f'{x:7.3f}')
        
        print("\nCorrelation Matrix (Multicollinearity):")
        print(corr_matrix.to_string())
    else:
        print("train_set.csv not found. Please ensure the stratification step ran successfully.")

    # --- XGBOOST MODEL TRAINING ---
    print("\n--- MODEL DEVELOPMENT (XGBoost) ---")
    try:
        import xgboost as xgb
        from sklearn.metrics import classification_report
        
        # Load the newly separated sets
        print("Loading Train, Validation, and Test sets for modeling...")
        train_set = pd.read_csv('train_set.csv')
        val_set = pd.read_csv('val_set.csv')
        test_set = pd.read_csv('test_set.csv')
        
        # Define columns we won't feed to the model
        cols_to_drop = ['Timestamp', 'Machine_ID', 'Efficiency_Status']
        
        # Prepare Training Data
        X_train = train_set.drop(columns=[c for c in cols_to_drop if c in train_set.columns])
        # Convert ordinal values (0.0, 0.5, 1.0) to integer classes (0, 1, 2)
        y_train = (train_set['Efficiency_Status'] * 2).astype(int)
        
        # Prepare Validation Data
        X_val = val_set.drop(columns=[c for c in cols_to_drop if c in val_set.columns])
        y_val = (val_set['Efficiency_Status'] * 2).astype(int)
        
        # Prepare Test Data
        X_test = test_set.drop(columns=[c for c in cols_to_drop if c in test_set.columns])
        y_test = (test_set['Efficiency_Status'] * 2).astype(int)
        
        # 1. Initialize the model
        print("Initializing XGBoost Classifier...")
        model = xgb.XGBClassifier(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=6, 
            objective='multi:softmax',
            num_class=3,
            early_stopping_rounds=10 # Modern way to handle early stopping
        )
        
        # 2. Train with Early Stopping
        print("Training model with Early Stopping (monitoring Validation set)...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 3. Predict and Evaluate
        print("\nPredicting on Validation set and Evaluating...")
        predictions = model.predict(X_val)
        
        # Print the final report
        target_names = ['Low (0.0)', 'Medium (0.5)', 'High (1.0)']
        print("\nClassification Report (Validation Set):")
        report_str = classification_report(y_val, predictions, target_names=target_names)
        print(report_str)
        
        # Save the report as a CSV with four decimal points
        report_dict = classification_report(y_val, predictions, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Save to CSV
        report_df.to_csv('Classification_Report.csv', float_format='%.4f')
        print("\nClassification report saved to 'Classification_Report.csv' with four decimal points.")
        
    except ImportError as e:
        print(f"\nLibrary Missing Error: {e}")
        print("Please resolve this by running the following in your terminal:")
        print("    pip3 install xgboost scikit-learn")
    except Exception as e:
        print(f"\nAn error occurred during model training: {e}")

