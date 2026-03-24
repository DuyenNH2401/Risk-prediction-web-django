import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import joblib

print("1. Loading 'train.csv'...")
df = pd.read_csv('../train.csv')

if 'id' in df.columns:
    df = df.drop(columns=['id'])

boolean_columns = ['road_signs_present', 'public_road', 'holiday', 'school_season']
categorical_columns = ['road_type', 'lighting', 'weather', 'time_of_day']
numeric_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

def bool_to_int(x):
    return x.astype(int)

def create_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('bool', FunctionTransformer(bool_to_int), boolean_columns),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns),
            ('num', 'passthrough', numeric_columns)
        ]
    )
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

X_full = df.drop(columns=['accident_risk'])
y_full = df['accident_risk']
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

train_data = X_train.copy()
train_data['accident_risk'] = y_train

print("2. Training Global Model...")
global_pipeline = create_pipeline()
global_pipeline.fit(X_train, y_train)

train_low = train_data[train_data['speed_limit'] <= 35].copy()
train_medium = train_data[(train_data['speed_limit'] > 35) & (train_data['speed_limit'] <= 55)].copy()
train_high = train_data[train_data['speed_limit'] > 55].copy()

print("3. Training Piecewise Models...")
low_pipeline = create_pipeline()
if not train_low.empty:
    low_pipeline.fit(train_low.drop(columns=['accident_risk']), train_low['accident_risk'])

medium_pipeline = create_pipeline()
if not train_medium.empty:
    medium_pipeline.fit(train_medium.drop(columns=['accident_risk']), train_medium['accident_risk'])

high_pipeline = create_pipeline()
if not train_high.empty:
    high_pipeline.fit(train_high.drop(columns=['accident_risk']), train_high['accident_risk'])

# Save to the local directory (since this script will be in the website folder)
joblib.dump({
    'global': global_pipeline,
    'low': low_pipeline,
    'medium': medium_pipeline,
    'high': high_pipeline
}, 'models_dict.pkl')
print("Successfully saved models_dict.pkl!")
