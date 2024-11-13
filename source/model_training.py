from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_model(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train,y_train)
    return model

from source.data_preprocessing import load_data, vectorize_data

df = load_data()
x_train, x_test, y_train, y_test = vectorize_data(df)
model = train_model(x_train,y_train)