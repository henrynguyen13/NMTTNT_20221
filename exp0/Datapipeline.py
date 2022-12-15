def print_score(labels, pred):
    from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
    print("Accuracy", accuracy_score(labels, pred))
    print("Precision", precision_score(labels, pred))
    print("Recall", recall_score(labels, pred))
    print("F1", f1_score(labels, pred))

def Datapipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = ['Payment', 'City_Tier', 'Gender', 'account_segment', 'Marital_Status', 'Login_device']
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    num_cols = ['Tenure', 'CC_Contacted_LY', 'Service_Score', 'Account_user_count','CC_Agent_Score', 'rev_per_month', 'Complain_ly','rev_growth_yoy', 'coupon_used_for_payment', 'Day_Since_CC_connect','cashback']
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    full_pp = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", DecisionTreeClassifier(random_state=42))
        ]
    )

    return full_pp

