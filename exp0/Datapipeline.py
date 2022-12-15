def print_score(labels, pred):
    from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
    print("Accuracy", accuracy_score(labels, pred))
    print("Precision", precision_score(labels, pred))
    print("Recall", recall_score(labels, pred))
    print("F1", f1_score(labels, pred))

def value_treatment(customers):
    import numpy as np
    customers['Gender'] = customers['Gender'].replace('F', 'Female')
    customers['Gender'] = customers['Gender'].replace('M', 'Male')
    customers['account_segment'] = customers['account_segment'].replace('Super +', 'Super Plus')
    customers['account_segment'] = customers['account_segment'].replace('Regular +', 'Regular Plus')

    customers['Tenure'] = customers['Tenure'].replace('#', np.nan)
    customers['Account_user_count'] = customers['Account_user_count'].replace('@', np.nan)
    customers['rev_per_month'] = customers['rev_per_month'].replace('+', np.nan)
    customers['rev_growth_yoy'] = customers['rev_growth_yoy'].replace('$', np.nan)
    customers['coupon_used_for_payment'] = customers['coupon_used_for_payment'].replace('#', np.nan)
    customers['coupon_used_for_payment'] = customers['coupon_used_for_payment'].replace('$', np.nan)
    customers['coupon_used_for_payment'] = customers['coupon_used_for_payment'].replace('*', np.nan)
    customers['cashback'] = customers['cashback'].replace('$', np.nan)
    customers['Day_Since_CC_connect'] = customers['Day_Since_CC_connect'].replace('$', np.nan)
    customers['Login_device'] = customers['Login_device'].replace('&&&&', 'Unknown')

    return customers

def Datapipeline(df):
    from sklearn.compose import ColumnTransformer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    customers = df.drop("AccountID", axis='columns')
    labels = customers['Churn']
    customers = customers.drop('Churn', axis='columns')

    customers = value_treatment(customers)

    customers_train, customers_val, labels_train, labels_val = train_test_split(customers, labels, test_size= 0.2, random_state=42, stratify=labels)

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

    full_pp.fit(customers_train, labels_train)
    pred_train = full_pp.predict(customers_train)
    pred_val = full_pp.predict(customers_val)

    print("---------------------------------------")
    print_score(labels_train, pred_train)
    print("---------------------------------------")
    print_score(labels_val, pred_val)
    return full_pp

