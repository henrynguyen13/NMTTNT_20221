import pandas as pd

def Datapipeline(df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = df.drop('Login_device', axis='columns')

    cat_cols = list(df.columns[df.columns.dtypes == object])
    # cat_cols = ['Payment', 'City_Tier', 'Gender', 'account_segment', 'Marital_Status', 'Login_device', 'Complain_ly']
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder())
        ]
    )

    num_cols = list(df.columns[df.columns[df.columns.dtypes != object]])
    # num_cols = ['Tenure', 'CC_Contacted_LY', 'Service_Score', 'Account_user_count','CC_Agent_Score', 'rev_per_month', 'rev_growth_yoy', 'coupon_used_for_payment', 'Day_Since_CC_connect','cashback']
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("dim_reduce", PCA(n_components=8)),
            ("scale", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    full_pp = Pipeline(
        steps=[
            ("preprocessor", preprocessor)
        ]
    )

    return full_pp

