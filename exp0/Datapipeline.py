def Datapipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler

    cat_cols = ['Payment', 'City_Tier', 'Gender', 'account_segment', 'Marital_Status', 'Login_device']
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder())
        ]
    )

    num_cols = ['Tenure', 'CC_Contacted_LY', 'Service_Score', 'Account_user_count','CC_Agent_Score', 'rev_per_month', 'Complain_ly','rev_growth_yoy', 'coupon_used_for_payment', 'Day_Since_CC_connect','cashback']
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
