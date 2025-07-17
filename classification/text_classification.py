import pandas as pd
import nltk
# Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def textClassification(url_path, file_path):
    from nltk.corpus import stopwords
    from sklearn.metrics import accuracy_score, classification_report

    # Contoh data dengan 3 label: 2 = Positif, 1 = Netral, 0 = Negatif
    texts = [
        "saya senang belajar",       # Positif
        "ini adalah bencana",        # Negatif
        "saya sangat senang",        # Positif
        "saya tidak suka ini",       # Negatif
        "saya merasa biasa saja",    # Netral
        "ini adalah hari yang baik", # Positif
        "saya tidak punya pendapat", # Netral
        "ini cukup membosankan"      # Negatif
    ]
    labels = [2, 0, 2, 0, 1, 2, 1, 0]  # Label sesuai dengan teks

    # Langkah praproses: vektorisasi dengan TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Bagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

    # Model SVM dengan kernel linear
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    print("Akurasi:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred, target_names=["Negatif", "Netral", "Positif"]))
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    path = file_path

    df = pd.read_csv(path)
    print(df)

    df['Ulasan_preprocessed'] = df['Ulasan_preprocessed'].fillna('')
    texts = df['Ulasan_preprocessed']

    print(df['label_sentiment'].value_counts())

    # Menentukan texts dan labels dari DataFrame
    texts = df['Ulasan_preprocessed']
    labels = df['label_sentiment']

    # Langkah praproses: Vektorisasi dengan TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Membagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

    # Melatih model SVM dengan kernel linear
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Melakukan prediksi dan evaluasi
    y_pred = model.predict(X_test)

    # Mencetak akurasi dan laporan klasifikasi
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))


    import pickle
    # Menentukan texts dan labels dari DataFrame
    texts = df['Ulasan_preprocessed'].fillna("") # Replace NaN with empty string
    labels = df['label_sentiment']

    # Langkah praproses: Vektorisasi dengan TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts.astype(str)) # Convert to string type explicitly

    # Membagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

    with open('classification/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Melatih model SVM dengan kernel linear
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Melakukan prediksi dan evaluasi
    y_pred = model.predict(X_test)

    # Mencetak akurasi dan laporan klasifikasi
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    filename = 'classification/svm_model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Load the saved model (optional, to verify saving)
    loaded_model = pickle.load(open(filename, 'rb'))

    # Example sentence
    new_text = ["Sangat baik"]

    # Vectorize the new text using the same vectorizer used to train the model
    # It is important that the vectorizer used is the same vectorizer that the model was trained on
    vectorizer = TfidfVectorizer() # Initialize a new TfidfVectorizer instance
    vectorizer = pickle.load(open('classification/vectorizer.pkl','rb')) # Load the vectorizer from the pickle file

    new_X = vectorizer.transform(new_text)


    # Make prediction using the loaded model
    prediction = loaded_model.predict(new_X)

    # Print the prediction
    print(prediction) # Output will be 0 or 1
    if prediction == 0:
        print("Netral")
    elif prediction == 1:
        print("Positif")
    else:
        print("Negatif")


    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # Pembagian data (gunakan stratifikasi jika diperlukan)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42, stratify=labels)

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    print("Logistic Regression")
    print("Akurasi:", accuracy_score(y_test, y_pred_log_reg))
    print(classification_report(y_test, y_pred_log_reg, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # 2. Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print("\nRandom Forest Classifier")
    print("Akurasi:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # 3. Naive Bayes Classifier (MultinomialNB)
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    print("\nNaive Bayes Classifier")
    print("Akurasi:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # 4. K-Nearest Neighbors (KNN)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    y_pred_knn = knn_clf.predict(X_test)
    print("\nK-Nearest Neighbors (KNN)")
    print("Akurasi:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # # Prediksi menggunakan setiap model
    # y_pred = model.predict(X_test)                 # SVM
    # y_pred_log_reg = log_reg.predict(X_test)           # Logistic Regression
    # y_pred_rf = rf_clf.predict(X_test)                 # Random Forest
    # y_pred_nb = nb_clf.predict(X_test)                 # Naive Bayes
    # y_pred_knn = knn_clf.predict(X_test)               # K-Nearest Neighbors

    # Membuat salinan dari DataFrame asli yang berisi data uji
    df_test = df.iloc[y_test.index].copy()  # Menggunakan index yang sama dengan data uji

    # Menambah kolom untuk setiap hasil prediksi model
    df_test['predicted_label_svm'] = y_pred
    df_test['predicted_label_log_reg'] = y_pred_log_reg
    df_test['predicted_label_rf'] = y_pred_rf
    df_test['predicted_label_nb'] = y_pred_nb
    df_test['predicted_label_knn'] = y_pred_knn

    # Menampilkan DataFrame dengan kolom baru
    print(df_test.head())

    # Menyimpan DataFrame ke file CSV jika diperlukan
    df_test.to_csv('classification/lama_hasil_prediksi_berbagai_model.csv', index=False)
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df_test.to_excel(writer, sheet_name='Ulasan_classification', index=False)