import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

def process_time_intervals(df):
    df = df.copy()

    def parse_sleep_duration(x):
        if pd.isnull(x): return np.nan
        x = str(x).lower()
        if '-' in x:
            parts = x.split('-')
            try:
                return (float(parts[0]) + float(parts[1].split()[0])) / 2
            except:
                return np.nan
        elif 'less' in x:
            return 4.5
        elif 'more' in x:
            return 9.5
        try:
            return float(x.split()[0])
        except:
            return np.nan

    def parse_work_hours(x):
        try:
            return float(x)
        except:
            return np.nan

    df['Sleep Duration'] = df['Sleep Duration'].apply(parse_sleep_duration)
    df['Work/Study Hours'] = df['Work/Study Hours'].apply(parse_work_hours)
    return df


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df = process_time_intervals(train_df)
test_df = process_time_intervals(test_df)

X = train_df.drop(columns=["Depression", "id", "Name"])
y = train_df["Depression"]
X_test = test_df.drop(columns=["id", "Name"])

numeric_features = ['Age', 'CGPA', 'Sleep Duration', 'Work/Study Hours',
                    'Academic Pressure', 'Work Pressure', 'Financial Stress',
                    'Study Satisfaction', 'Job Satisfaction']

categorical_features = ['Gender', 'City', 'Working Professional or Student',
                        'Profession', 'Dietary Habits', 'Degree',
                        'Have you ever had suicidal thoughts ?',
                        'Family History of Mental Illness']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42))
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print("F1-score на валидации:", f1_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

model.fit(X, y)
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "id": test_df["id"],
    "Depression": test_pred
})
submission.to_csv("submission.csv", index=False)