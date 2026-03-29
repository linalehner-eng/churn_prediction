# imports
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.utils.multiclass import unique_labels

# load data
dir_path = os.path.dirname(os.path.realpath(__file__))
df_train = pd.read_csv(os.path.join(dir_path, 'data', 'raw', 'train.csv'))
df_test = pd.read_csv(os.path.join(dir_path, 'data', 'raw', 'test.csv'))

# drop target column 'Churn'
X_train = df_train.drop('Churn', axis=1)
y_train = df_train['Churn']
X_test = df_test

categorical_features = [ 'gender', 'Partner', 'Dependents',  'PhoneService', 'MultipleLines', 'InternetService', 'DeviceProtection', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
ordinal_features = ['Contract']
metric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# convert strings to numeric values
# get unique values from each categorical feature
cat_unique_labels = [unique_labels(X_train[column]) for column in categorical_features]

# get unique values from contract
contract_unique_labels = unique_labels(X_train['Contract'])

# Create a ColumnTransformer to apply transformations only to specific columns.
# 'remainder="passthrough"' means other columns will be left unchanged.
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(categories=cat_unique_labels), categorical_features),
        ('ordinal', OrdinalEncoder(categories=[contract_unique_labels]), ordinal_features)
    ],
    remainder='passthrough'
)

# Fit and transform the features
X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.fit_transform(X_test)

# Transform the target using LabelEncoder (or another appropriate encoder)
target_encoder = LabelEncoder()
y_train_transformed = target_encoder.fit_transform(y_train)

# Convert to DataFrame with appropriate column names
transformed_column_names_X_train = column_transformer.get_feature_names_out()
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=transformed_column_names_X_train)

transformed_column_names_X_test = column_transformer.get_feature_names_out()
X_test_transformed_df = pd.DataFrame(X_train_transformed, columns=transformed_column_names_X_test)

# Add the transformed target as a new column
X_train_transformed_df['Churn'] = y_train_transformed

# Save data
X_train_transformed_df.to_csv('data/processed/X_train_trans.csv', index=False)
X_test_transformed_df.to_csv('data/processed/X_test_trans.csv', index=False)



