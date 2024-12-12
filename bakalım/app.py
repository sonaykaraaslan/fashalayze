import os
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Veri dosyalarının yolu
positive_path = 'data/Train_Set_Positive_Processed.txt'
negative_path = 'data/Train_Set_Negative_Processed.txt'

# Verileri okuma
with open(positive_path, 'r', encoding='utf-8') as file:
    positive_data = file.readlines()

with open(negative_path, 'r', encoding='utf-8') as file:
    negative_data = file.readlines()

# Etiketleme
positive_labels = [1] * len(positive_data)  # 1: pozitif etiket
negative_labels = [0] * len(negative_data)  # 0: negatif etiket

# Veriyi birleştirme
data = positive_data + negative_data
labels = positive_labels + negative_labels

# Metin verisini sayısallaştırma
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data)

# Modeli eğitme
model = MultinomialNB()
model.fit(X, labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    
    return render_template('index.html', sentiment=sentiment, text=text)

if __name__ == "__main__":
    app.run(debug=True)
