import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# import a library for lable encoding
from sklearn.preprocessing import LabelEncoder

#import library for train test split
from sklearn.model_selection import train_test_split
#importing the libraries for precision matrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import confusion_matrix
df=pd.read_csv("Titanic-Dataset.csv")
df.head()
df.shape
x=df[["Pclass","Age","Sex","SibSp","Parch"]]
y=df["Survived"]
x["Age"]=x["Age"].fillna(x["Age"].mean())
x.info()
encoder=LabelEncoder()
x["Sex"]=encoder.fit_transform(x["Sex"])
x.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#dictionary of models for model setup
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
#training and evaluting the model
results=[]

for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    print("Classification report of all the machine learning algorithm")
    cm=confusion_matrix(y_test,y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    results.append(
        {
            "Model Name": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    )
    results_df = pd.DataFrame(results)
    # Visulaize the confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

  # Visual the result
    plt.figure(figsize=(12, 8))
    results_df.set_index("Model Name")[["Accuracy", "Precision", "Recall"]].plot(kind="bar", cmap="magma")
    plt.title("Visulization of Model Performance")
    plt.show()

    #summary  of the model

print("Summary of models")
print(results_df)
import joblib
joblib.dump(model,'titanic_model.pkl')
joblib.dump(encoder,'sex_encoder.pkl')