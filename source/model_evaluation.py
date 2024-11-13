from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))