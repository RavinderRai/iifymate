import pandas as pd
from typing import Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def remove_stop_words(input_string: str) -> str:
    english_stop_words = stopwords.words('english')

    #get the words in the review as a list
    input_words = input_string.split()
    
    #make a new list with the same words but only if they are not a stop word
    removed_stop_words_list = [word for word in input_words if word not in english_stop_words]
    
    removed_stop_words = ' '.join(removed_stop_words_list)
    
    return removed_stop_words

def lemmatizing(input_string: str) -> str:
    lemmatizer = WordNetLemmatizer()

    #get the words in the input string as a list
    input_words = input_string.split()
    
    #lemmatize the words
    lemmatized_list = [lemmatizer.lemmatize(word) for word in input_words]
    
    #make it into a string again
    lemmatized_string = ' '.join(lemmatized_list)
    
    return lemmatized_string

def get_tfidf_splits(
    X: pd.Series, 
    y: pd.Series, 
    test_size: float = 0.25, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, TfidfVectorizer]:
    """
    Split data into training and testing sets and perform TF-IDF vectorization.

    Parameters
    ----------
    X : pd.Series
        Input DataFrame containing recipe data.
    y : pd.Series
        Target variable.
    test_size : float, optional
        Size of the testing set. Defaults to 0.25.
    random_state : int, optional
        Random seed for reproducibility. Defaults to 42.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target variable.
    y_test : pd.Series
        Testing target variable.
    tfidf_fitted : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    tfidf = TfidfVectorizer()

    tfidf_fitted = tfidf.fit(X_train.str.join(' '))

    tfidf_X_train_labels = tfidf_fitted.transform(X_train.str.join(' '))
    tfidf_X_test_labels = tfidf_fitted.transform(X_test.str.join(' '))
    tfidf_train_df = pd.DataFrame(tfidf_X_train_labels.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_test_df = pd.DataFrame(tfidf_X_test_labels.toarray(), columns=tfidf.get_feature_names_out())

    return tfidf_train_df, tfidf_test_df, y_train, y_test, tfidf_fitted

def SVD_reduction(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, TruncatedSVD]:
    """
    Perform SVD reduction on the given data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Input DataFrame containing the training set.
    X_test : pd.DataFrame
        Input DataFrame containing the testing set.
    n_components : int, optional
        Number of components to keep. Defaults to 1000.

    Returns
    -------
    X_train_reduced_df : pd.DataFrame
        DataFrame containing the reduced training set.
    X_test_reduced_df : pd.DataFrame
        DataFrame containing the reduced testing set.
    svd_fitted : TruncatedSVD
        Fitted SVD object.
    """
    svd = TruncatedSVD(n_components=n_components)
    svd_fitted = svd.fit(X_train)
    X_train_reduced, X_test_reduced = svd.transform(X_train), svd.transform(X_test)

    #getting column names just to convert to dataframe
    column_names = [f"component_{i+1}" for i in range(X_train_reduced.shape[1])]
    X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=column_names, index=X_train.index)
    X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=column_names, index=X_test.index)

    return X_train_reduced_df, X_test_reduced_df, svd_fitted
