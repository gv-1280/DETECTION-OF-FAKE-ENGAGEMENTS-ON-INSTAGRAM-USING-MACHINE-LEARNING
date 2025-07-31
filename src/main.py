#data handling 
import pandas as pd
import numpy as np
import os 

#for data visualization
import matplotlib.pyplot as plt

#for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix, classification_report,ConfusionMatrixDisplay

#for model saving
import joblib

current_dir = os.path.dirname(os.path.abspath('data/custom_instagram_dataset.csv'))
project_dir = os.path.dirname(current_dir)
data_folder_path = os.path.join(project_dir,'data')
dataset_file_path = os.path.join(data_folder_path,'custom_instagram_dataset.csv')
print("attempting to load dataset from :" ,{dataset_file_path})

#load dataset using os to make universal
try :
    df = pd.read_csv(dataset_file_path)
   
    print("Dataset is sucessfully loaded")
    print(df.head())
except FileNotFoundError:
    print("Not found ")
except Exception as e:
    print("Unexpected error accur while loading the file ")


df['label'] = df['label'].map({'real' : 0,'fake' : 1})
#print(df.dtypes) #to check if the dataset is in numeric value or not 
selected_features = [
    'follower_count',
    'like_count',
    'comment_count',
    'avg_comment_length',
    'emoji_comment_ratio',
    'like_comment_ratio'  ]
x = df[selected_features]
y = df['label']
# print(x.head())
# print(x.shape)

#splitting the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#create and train the model
rf = RandomForestClassifier(n_estimators=100,random_state=42)
lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(use_label_encoder=False,eval_matric='logloss')

#make a ensemble classifier 
ensemble = VotingClassifier(
    estimators= [('rf', rf),('lr', lr),('xgb', xgb)],
    voting='hard'
)

#train your model
ensemble.fit(x_train,y_train)
print("x_train shape ",x_train.shape)
print("feature used for training the model ",x_train.columns)
print(df['label'].value_counts())

#make predictions
y_pred = ensemble.predict(x_test)

# rf_model = ensemble.named_estimators_['rf']
# importances = rf_model.feature_importances_
# for name, score in zip(x.columns, importances):
#     print(f"{name}: {score:.4f}")

#checking accuracy
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_pred,y_test)
print("Accuracy : ",accuracy) 

#to check where the model is making errors
cm = confusion_matrix(y_test,y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()

#classification report
print(classification_report(y_test,y_pred))

#feature importance
rf_model = ensemble.named_estimators_['rf']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = np.array(x.columns)

plt.figure(figsize=(10,6))
plt.title("Feature importances")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

#save the model
try:
    joblib.dump(ensemble,'models/ensemble_model.pkl')
    print("the model is saved successfully")
except Exception as e:
    print("Caused an issue ",e)
