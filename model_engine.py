from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
def create_pipeline():
    numeric_features = ['Eccentricity', 'Inclination', 'Mean Motion', 'Epoch', 'Altitude']
    return Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)])),
        ('regressor', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
    ])