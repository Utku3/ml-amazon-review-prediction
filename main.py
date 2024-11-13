from source.data_preprocessing import load_data, vectorize_data
from source.model_training import train_model
from source.model_evaluation import evaluate_model

df = load_data()
x_train, x_test, y_train, y_test = vectorize_data(df)
model = train_model(x_train, y_train)
evaluate_model(model, x_test, y_test)