from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
app = Flask(__name__)
model = None
tokenizer = None
analyzer = SentimentIntensityAnalyzer()

def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')


def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

load_keras_model()
load_tokenizer()

def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(
        user_sequences,
        maxlen=1225
    )
    prediction = model.predict(user_sequences_matrix, verbose=0)
    return round(float(prediction[0][0]), 2)

        
@app.route("/", methods=["GET", "POST"])
def index():
    # TODO: Write the code that calls the sentiment analysis functions here.
    # hint: use request.method == "POST"
    sentiment = None

    if request.method == "POST":
        text = request.form.get("user_text")
        sentiment = analyzer.polarity_scores(text)
        sentiment["custom model positive"] = sentiment_analysis(text)

        if sentiment["compound"] >= 0.05:
            sentiment["label"] = "Positive"
        elif sentiment["compound"] <= -0.05:
            sentiment["label"] = "Negative"
        else:
            sentiment["label"] = "Neutral"

    return render_template('form.html', sentiment=sentiment)
if __name__ == "__main__":
    app.run()
