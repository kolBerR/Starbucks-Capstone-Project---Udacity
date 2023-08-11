import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# read the wrangled csv data from the data folder
final_data= pd.read_csv('data/final_data.csv')


# initialize a standard scaler object from the sklearn.preprocessing module
scaler = StandardScaler()
# instantiate a classifier based on sklearn logistic regression model
lrc = LogisticRegression(random_state=42, solver='liblinear')
# initialize another classifier based on the random forest model
rfc = RandomForestClassifier(random_state=42)



# drop columns not needed for the models
df = final_data.drop(columns=['offer_id', 'time', 'customer_id', 'duration', 'became_member_on'])
# create a list of features to scale
features = ['total_amount', 'reward', 'difficulty', 'duration_in_hrs']

def scaling_function(df, features):
    
    """
    A function to scale a list of features
    
    INPUT:
        df (dataframe): a dataframe
        features (list): a selected list of columns to apply the scale function

    OUTPUT:
        df_scaled(dataframe): a dataframe recombining back the scaled features with the other variables
    """
        
    # a dataframe formed by extracting the features selected for scaling
    cols_to_scale = df[features]
        
    # apply the scaler's fit_transform method
    scaled_features = pd.DataFrame(scaler.fit_transform(cols_to_scale), columns = cols_to_scale.columns, index=cols_to_scale.index)
        
    # remove the orignal features from the dataframe 
    df = df.drop(columns=features, axis=1)

    # recombine the scaled features with the input dataframe
    df_scaled = pd.concat([df, scaled_features], axis=1)
        
    return df_scaled

# X includes the independent variables that act as explanaroty variables in the model
X = scaling_function(df=df, features=features).drop(columns=['offer_completed'])

# y holds data on the feature we will try to predict
y = scaling_function(df=df, features=features)['offer_completed']

# split data into train and test sets using sklearn's train_test_split() method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def apply_model():
    #train model using RandomForestClassifier
    rfc.fit(X_train, y_train)
    # retrieve the trained parameters from the RandomForestClassifier
    explanatory_pwr = pd.DataFrame(rfc.feature_importances_,index=X_train.columns.tolist(),columns=['exp_pwr']).reset_index()
    # rename the index
    explanatory_pwr.rename(columns={'index': 'feature'}, inplace=True)
    # calculate the explanatory power of each variable  
    explanatory_pwr['exp_pwr_strength'] = np.round((explanatory_pwr['exp_pwr']/explanatory_pwr['exp_pwr'].sum())*100,2)
    # sort the variables
    explanatory_pwr = explanatory_pwr.sort_values(by=['exp_pwr_strength'], ascending=False).reset_index(drop=True)
    # drop the 'exp_pwr' variable
    explanatory_pwr.drop(columns=['exp_pwr'],inplace=True)

    # predict using the model:
    y_pred_test = rfc.predict(X_test)
    # generate confusion matrix
    matrix = confusion_matrix(y_test, y_pred_test)

    # x - values
    x = explanatory_pwr.loc[0:4, 'exp_pwr_strength'].values.tolist()
    # y - features
    y = explanatory_pwr.loc[0:4, 'feature'].values.tolist()


    return matrix, x, y




