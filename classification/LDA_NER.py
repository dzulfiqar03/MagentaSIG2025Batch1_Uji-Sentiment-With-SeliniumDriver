from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def LDAdanNER(url_path):
    # Daftar kata-kata positif dan negatif
    positive_words = ["bagus", "baik", "indah", "positif", "sukses"]
    negative_words = ["buruk", "jelek", "negatif", "gagal", "sedih"]

    def classify_sentiment_lexicon(text):
        words = text.split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        if positive_count > negative_count:
            return 'positif'
        elif positive_count < negative_count:
            return 'negatif'
        else:
            return 'netral'

    def calculate_coherence(topics, dt_matrix):
        """
        Simple coherence: mean pairwise cosine similarity between top words in topic
        """
        scores = []
        for topic in topics:
            indices = [feature_names.tolist().index(w) for w in topic if w in feature_names]
            if len(indices) >= 2:
                vecs = dt_matrix[:, indices].T.toarray()
                sim_matrix = cosine_similarity(vecs)
                upper_tri = sim_matrix[np.triu_indices(len(indices), k=1)]
                score = np.mean(upper_tri) if len(upper_tri) else 0
                scores.append(score)
            else:
                scores.append(0)
        return np.mean(scores)

    # Load dataset
    df = pd.read_csv(url_path, encoding='utf-8')
    df = df.dropna(subset=['Ulasan_preprocessed'])
    df['sentimen'] = df['Ulasan_preprocessed'].apply(classify_sentiment_lexicon)

    stopword_factory = StopWordRemoverFactory()
    stopwords_id = stopword_factory.get_stop_words()

    coherence_scores = []
    num_topics_range = range(1, 4)

    for sentiment in ['positif', 'netral', 'negatif']:
        print(f"\nAnalisis untuk sentimen: {sentiment.upper()}")
        subset = df[df['sentimen'] == sentiment]['Ulasan_preprocessed']

        if subset.empty:
            print(f"Tidak ada data untuk sentimen '{sentiment}', dilewati.")
            continue

        subset = subset[subset.str.strip().astype(bool)]

        if len(subset) == 0:
            print(f"Tidak ada ulasan yang valid setelah preprocessing untuk sentimen '{sentiment}', dilewati.")
            continue

        vectorizer = CountVectorizer(stop_words=stopwords_id)
        doc_term_matrix = vectorizer.fit_transform(subset)

        feature_names = vectorizer.get_feature_names_out()

        for num_topics in num_topics_range:
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(doc_term_matrix)

            topics = lda.components_
            topic_words = [
                [feature_names[i] for i in topic.argsort()[-10:]]
                for topic in topics
            ]

            coherence = calculate_coherence(topic_words, doc_term_matrix)
            coherence_scores.append((sentiment, num_topics, coherence))

            if num_topics == 1:
                topic = lda.components_[0]
                topic_words_freq = {feature_names[i]: topic[i] for i in topic.argsort()[-100:]}
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words_freq)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f"Word Cloud untuk Sentimen {sentiment.capitalize()}")
                plt.axis('off')
                plt.show()

    plt.figure(figsize=(10, 6))
    for sentiment in ['positif', 'netral']:
        scores = [score for s, num_topics, score in coherence_scores if s == sentiment]
        plt.plot(num_topics_range, scores, marker='o', label=f"Sentimen {sentiment.capitalize()}")

    plt.xlabel("Jumlah Topik")
    plt.ylabel("Nilai Coherence (versi sederhana)")
    plt.title("Grafik Coherence Score untuk Setiap Sentimen")
    plt.legend()
    plt.grid()
    plt.show()
