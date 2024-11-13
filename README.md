# ml-imdb-review-prediction

This project uses machine learning to predict if an IMDb review is positive or negative.

The dataset of the reviews are available [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). It contains reviews labeled with a sentiment (either positive or negative). The data was split 80/20 for training and testing respectively.

    Classification Report:
               
                precision   recall   f1-score   support
    negative       0.90      0.88      0.89      4961
    positive       0.88      0.91      0.89      5039

    accuracy                           0.89     10000
    macro avg      0.89      0.89      0.89     10000
    weighted avg   0.89      0.89      0.89     10000


    Confusion Matrix:

    [[4349  612]
    [ 478 4561]]
These results show that the model we trained is able to correctly predict the sentiment of the IMDb reviews in our dataset with 89% accuracy.