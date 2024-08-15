
import pandas as pd

file_path = '/content/Vegetable-fruits-Prices-2022.csv'

data = pd.read_csv(file_path, encoding='latin-1')

data.dropna(axis=1, how='all', inplace=True)

data.drop_duplicates(inplace=True)

data.fillna(method='ffill', inplace=True)

cleaned_file_path = '/content/Vegetable-fruits-Prices-2022-cleaned.csv'
data.to_csv(cleaned_file_path, index=False)

print(f"Datos limpiados y guardados en '{cleaned_file_path}'")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


file_path = '/content/Vegetable-fruits-Prices-2022.csv'

data = pd.read_csv(file_path, encoding='latin-1')

data.dropna(axis=1, how='all', inplace=True)
data.drop_duplicates(inplace=True)
data.fillna(method='ffill', inplace=True)

encoder = OneHotEncoder(handle_unknown='ignore')
encoded_form = encoder.fit_transform(data[['Form']])
encoded_form_df = pd.DataFrame.sparse.from_spmatrix(encoded_form, columns=encoder.get_feature_names_out(['Form']))
data = pd.concat([data, encoded_form_df], axis=1)
data = data.drop('Form', axis=1)

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(data[[col]])
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=encoder.get_feature_names_out([col]))
    data = pd.concat([data, encoded_df], axis=1)
    data = data.drop(col, axis=1)


print(data.columns)

data['RetailPrice'] = data[['RetailPrice']].mean(axis=1)


cleaned_file_path = '/content/Vegetable-fruits-Prices-2022-cleaned.csv'
data.to_csv(cleaned_file_path, index=False)

features = data.drop('RetailPrice', axis=1)
target = data['RetailPrice']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Datos limpiados y guardados en '{cleaned_file_path}'")
