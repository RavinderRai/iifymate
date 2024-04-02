from flask import Flask, request, jsonify
import pickle
import os
from google.cloud import storage
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

def remove_stop_words(review):
    english_stop_words = stopwords.words('english')

    #get the words in the review as a list
    review_words = review.split()
    
    #make a new list with the same words but only if they are not a stop word
    removed_stop_words_list = [word for word in review_words if word not in english_stop_words]
    
    removed_stop_words = ' '.join(removed_stop_words_list)
    
    return removed_stop_words

def lemmatizing_reviews(review):
    lemmatizer = WordNetLemmatizer()

    #get review text as a list of words
    review_list = review.split()
    
    #lemmatize the words
    lemmatized_list = [lemmatizer.lemmatize(word) for word in review_list]
    
    #make it into a string again
    lemmatized_review = ' '.join(lemmatized_list)
    
    return lemmatized_review

def load_artifact_from_gcs(artifact_path, bucket_name='macro_predictor'):
    """
    Loads an artifact from a pickle file stored in Google Cloud Storage.
    Args:
        bucket_name (str): Name of the GCS bucket.
        artifact_path (str): Path within the bucket where the artifact is stored.
    Returns:
        object: Loaded pickle file.
    """
    # Initialize a client and bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Get the blob (pickle file) from GCS
    blob = bucket.blob(artifact_path)

    # Download the pickle file contents as bytes
    pickle_bytes = blob.download_as_string()

    # Load the dictionary from the downloaded pickle file contents
    loaded_artifact = pickle.loads(pickle_bytes)

    return loaded_artifact

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.get_json(force=True)
    user_input = user_input['user_input']

    gcp_config_file = '../flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file

    svd_fitted = load_artifact_from_gcs('macro_data_processing/svd_fitted.pkl')
    tfidf_fitted = load_artifact_from_gcs('macro_data_processing/tfidf_fitted.pkl')

    XGBoost_fat_model = load_artifact_from_gcs('training/XGBoost_fat_model.pkl')
    XGBoost_carbs_model = load_artifact_from_gcs('training/XGBoost_carbs_model.pkl')
    XGBoost_protein_model = load_artifact_from_gcs('training/XGBoost_protein_model.pkl')

    user_input = remove_stop_words(user_input)
    user_input = lemmatizing_reviews(user_input)
    user_input = tfidf_fitted.transform([user_input])
    user_input = svd_fitted.transform(user_input)

    predicted_fat = int(np.expm1(XGBoost_fat_model.predict(user_input)[0]))
    predicted_carbs = int(np.expm1(XGBoost_carbs_model.predict(user_input)[0]))
    predicted_protein = int(np.expm1(XGBoost_protein_model.predict(user_input)[0]))

    calories = 9*predicted_fat + 4*(predicted_carbs + predicted_protein)
    
    return jsonify({
        'predicted_fat': predicted_fat,
        'predicted_carbs': predicted_carbs,
        'predicted_protein': predicted_protein,
        'calories': calories
    })



if __name__ == '__main__':
    app.run(debug=True)