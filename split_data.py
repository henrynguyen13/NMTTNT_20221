from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

cur_dir = os.getcwd()
customers = pd.read_csv(cur_dir + "/original_data/CustomerChurnData.csv")
customers = customers.drop("AccountID", axis='columns')

#BEGIN//replace all special values to NaN
customers['Tenure'] = customers['Tenure'].replace('#', np.nan)

customers['Gender'] = customers['Gender'].replace('F', 'Female')
customers['Gender'] = customers['Gender'].replace('M', 'Male')

customers['Account_user_count'] = customers['Account_user_count'].replace('@', np.nan)

customers['account_segment'] = customers['account_segment'].replace('Regular +', 'Regular Plus')
customers['account_segment'] = customers['account_segment'].replace('Super +', 'Super Plus')

customers['rev_per_month'] = customers['rev_per_month'].replace('+', np.nan)

customers['rev_growth_yoy'] = customers['rev_growth_yoy'].replace('$', np.nan)


customers['coupon_used_for_payment'] = customers['coupon_used_for_payment'].replace('#', np.nan)
customers['coupon_used_for_payment'] = customers['coupon_used_for_payment'].replace('$', np.nan)
customers['coupon_used_for_payment'] = customers['coupon_used_for_payment'].replace('*', np.nan)

customers['Day_Since_CC_connect'] = customers['Day_Since_CC_connect'].replace('$', np.nan)

customers['cashback'] = customers['cashback'].replace('$', np.nan)

customers['Login_device'] = customers['Login_device'].replace('&&&&', np.nan)
#END//replace all special values to NaN

customers_train, customers_test = train_test_split(customers, test_size= 0.2, random_state=42, stratify=customers['Churn'])
customers_test = pd.DataFrame(customers_test)
customers_train.to_csv(cur_dir + '/exp/train.csv', index = False)
customers_test.to_csv(cur_dir + '/exp/test.csv', index=False)

