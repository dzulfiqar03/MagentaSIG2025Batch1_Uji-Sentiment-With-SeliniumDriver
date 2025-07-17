import pandas as pd
import nltk
# Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import data
def classification_preprocessing(url_path, df_ulasan, df_produk):
    from nltk.tokenize import word_tokenize

    nltk.downloader.download('vader_lexicon')
    path = url_path


    # Baca file Excel
    df = pd.read_excel(path)
    df = df[df['Perusahaan'] == "Netafarm"]
    # Tampilkan seluruh isi DataFrame
    print(df)

    # Contoh: membaca isi kolom 'Ulasan' (ubah sesuai nama kolom yang ada di Excel kamu)
    print(df['Ulasan'])
    
    
    
    import gdown
    import json

    # Function to download and load slang dictionary from Google Drive
    def load_slang_dict(drive_url):
        # Extract the file ID from the Google Drive URL
        file_id = drive_url.split('/')[-2]

        # Construct the download URL
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

        # Download the file using gdown (it will be saved locally as 'slang_dict.txt')
        gdown.download(download_url, 'classification/slang_dict.txt', quiet=False)

        # Now open the downloaded file and load it as a dictionary
        with open('classification/slang_dict.txt', 'r', encoding='utf-8') as file:
            slang_dict = json.load(file)

        return slang_dict

    # Example usage:
    drive_url = 'https://drive.google.com/file/d/1lzuTJRSVOuQd1GEO7HjOfX0tEWQvqT3t/view?usp=drive_link'
    slang_dict = load_slang_dict(drive_url)

    # Print the loaded slang dictionary
    print(slang_dict)

    new_slang_dict = {
    # Kata slang untuk "tidak"
    "👍": "bagus",
    "👍🏻": "bagus","good": "Baik",
    "👌": "bagus","👌👌": "bagus","👌🏻": "bagus","bagusbagusbagus": "bagus","sngat bagus": "Sangat Bagus",
    "😂": "Senyum semangat",
    "😅": "senyum kecil",
    "🥵": "aduh",
    "👎": "jelek",
    "😒👎": "jelek",
    "🤦‍♂️": "aduh",
    "🤦🏼‍♂️": "aduh",
    "👊": "hantam",
    "trims": "terima kasih",
    "g": "tidak",
    "❤️": "aplikasi favorit",
    "🥰": "aplikasi favorit",
    "😍": "aplikasi favorit",
    "🥰😘": "aplikasi sangat favorit",
    "🥵🥵": "Jelek","😡": "Marah",
    "🥰🥰🥰🥰🥰🥰🥰🥰🥰🥰🥰🥰🥰": "aplikasi sangat amat favorit",
    "❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️": "aplikasi sangat amat favorit",
    "🥰😍😍": "aplikasi sangat favorit",
    "🤩🤩🤩🤩🤩": "aplikasi top",
    "🤗": "cukup bagus",
    "apik": "bagus",
    "𝐾𝑛𝑎𝑝𝑎ℎ 𝑔𝑘 𝑏𝑖𝑠𝑎 𝑚𝑎𝑠𝑢𝑘 𝑎𝑝𝑘 𝑛𝑦𝑎": "Kenapa tidak bisa masuk aplikasinya",
    "𝘔𝘯𝘵𝘢𝘱": "Mantap","𝐌𝐚𝐮 𝐝𝐚𝐟𝐭𝐚𝐫 𝐚𝐣𝐚 𝐞𝐫𝐫𝐨𝐫": "Mau daftar saja error",
    "wis": "sudah",
    "𝘗𝘢𝘳𝘢𝘩 𝘨𝘢 𝘣𝘪𝘴𝘢 𝘭𝘰𝘨𝘪𝘯.. 𝘈𝘴𝘶𝘶𝘶... 😩😩": "Parah, tidak bisa login, aduhh",
    "🤫🧏": "Diam",
    "👍👍👍😬": "bagus",
    "🤜🏻🤛🏻": "Sepakat",
    "anjg": "anjing",
    "💧🤢": "Muak",
    "🥳" : "sangat menikmati",
    "Sangat Memuaskan" : "sangat puas",
    "🤩" : "favorit",
    "🙈" : "Malu",
    "🤝" : "terima kasih",
    "✅✅✅✅" : "Aplikasi Sukses",
    "🗿👍" : "Cukup Bagus",
    "👍👍" : "Bagus",
    "👍🤟" : "Bagus keren",
    "👍👍👍" : "Sangat Bagus",
    "👍👍👍👍" : "Sangat Bagus",
    "👍👍👍👍👍" : "Sangat Bagus",
    "👍🏻👍🏻👍🏻👍🏻" : "Sangat Bagus",
    "👍👍👍👍👍👍" : "Sangat Bagus",
    "👍👍👍👍👍👍👍" : "Sangat Amat Bagus",
    "🌟🌟🌟🌟🌟" : "Bintang 5",
    "🔥🔥🔥🔥🔥🔥" : "Menyala",
    "😃😃😃😃" : "Menyenangkan",
    "☺️☺️☺️" : "Sangat Membantu","𝑠𝑎𝑛𝑔𝑔𝑎𝑡 𝑚𝑒𝑚𝑎𝑛𝑡𝑢" : "Sangat Membantu",
    "👀" : "Mengintip",
    "🌠" : "Semakin Lebih baik",
    "apk" : "aplikasi","𝐌𝐚𝐮 𝐝𝐚𝐟𝐭𝐚𝐫 𝐚𝐣𝐚 𝐞𝐫𝐫𝐨𝐫" : "Mau Daftar saja error",
    "✌️🥶" : "Aplikasi cetar",
    "😁😁😁" : "tertawa",
    "🤲👍" : "Semoga semakin lebih baik",
    ",i" : "ih bagus","🥲" : "kecewa",
    "🙏👍👍" : "Bagus",
    # Kata slang untuk "tidak"
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "g": "tidak",
    "kagak": "tidak",
    "enggak": "tidak",
    "mantul": "mantap betul"
    }
    slang_dict.update(new_slang_dict)

    def normalize_slang(text, slang_dict):
        words = word_tokenize(text)
        normalized_words = [slang_dict.get(word, word) for word in words]  # Replace slang if found in dictionary
        return ' '.join(normalized_words)
    # Text preprocessing function including tokenization, lowercasing, punctuation removal, stopword removal, slang normalization, and stemming
    def preprocess_text(text, slang_dict, stemmer):

        # Lowercasing
        text = text.lower()

        # Normalize slang and abbreviations
        text = normalize_slang(text, slang_dict)

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        custom_stopwords = set(stopwords.words('indonesian')).union(set(stopwords.words('english')))
        exclude_words = ['baik']
        custom_stopwords = custom_stopwords.difference(exclude_words)
        tokens = [word for word in tokens if word not in custom_stopwords]

        # Stemming
        tokens = [stemmer.stem(word) for word in tokens]

        # Join the tokens back into a single string
        return ' '.join(tokens)


    # Create a stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Example text for preprocessing
    text = "saya ingin paket internet unlimited lebih murah dari harga konter."

    # Preprocess the text
    preprocessed_text = preprocess_text(text, slang_dict, stemmer)

    # Print the final preprocessed text
    print(preprocessed_text)

    preprocessed_texts = []
    for index, row in df.iterrows():
        preprocessed_text = preprocess_text(row['Ulasan'], slang_dict, stemmer)
        preprocessed_texts.append(preprocessed_text)

    # Add the preprocessed text as a new column in the DataFrame
    df['Ulasan_preprocessed'] = preprocessed_texts

    # Print the result
    print(df[['Ulasan', 'Ulasan_preprocessed']])

    df.to_csv('classification/data_preprocessing.csv')


    from nltk.probability import FreqDist
    from nltk.tokenize import word_tokenize

    # Ensure the necessary NLTK data is downloaded
    nltk.download('punkt')

    all_words = []

    for index, row in df.iterrows():
        tokens = word_tokenize(row['Ulasan_preprocessed'])
        all_words.extend(tokens)  # Add all tokens from each document to a single list

    # Use NLTK's FreqDist to count word frequencies
    word_freq = FreqDist(all_words)

    # Get the most common words as a list of tuples (word, frequency)
    most_common_words = word_freq.most_common()

    # Convert the list of tuples into a pandas DataFrame
    df_word_freq = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

    # Save the DataFrame to a CSV file
    df_word_freq.to_csv('classification/word_frequencies.csv', index=False)

    # Print confirmation
    print("Word frequencies saved to 'classification/word_frequencies.csv'")

    def clean_text(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub(r'@\w+\s*', '', text) #hapus mention
        text = re.sub(r'https?://\S+', '', text) #hapus link
        text = re.sub('\[.*?\]', ' ', text)
        text = re.sub('\(.*?\)', ' ', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\w*\d\w*', ' ', text)
        '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
        text = re.sub('[‘’“”…♪♪]', '', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('\xa0', ' ', text)
        text = re.sub('b ', ' ',text)
        text = re.sub('rt ', ' ', text)
        text = re.sub(r'\s+', ' ', text) #hapus spasi
        return text

    clean = lambda x: clean_text(x)


    #load kamus senti indo (lexicon berbahasa indonesia dari sini)
    url = 'https://drive.google.com/file/d/1qPX0Uej3PqUQUI3op_oeEr8AdmrgOT2V/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]

    df_senti = pd.read_csv(path, sep=':', names=['word', 'value'])

    senti_dict = {}

    for i in range (len(df_senti)):
        senti_dict[df_senti.iloc[i]['word']] = df_senti.iloc[i]['value']

    import reprlib
    senti_indo = SentimentIntensityAnalyzer()
    senti_indo.lexicon.update(senti_dict)

    #cek mengetahui apakah sautu kata ada pada lexicon dan cara mengetahui nilai suatu kata pada lexicon
    word = "negatif"
    if word in senti_indo.lexicon:
        print("Kata ditemukan dalam lexicon dengan skor:", senti_indo.lexicon[word])
    #catatan: senti_indo.lexicon[word] digunakan untuk mencri nilai lexicon suatu kata
    else:
        print("Kata  tidak ditemukan dalam lexicon.")
        
        #menghitung score positif, negatif, netral, dan nilai compound
    tokens = word_tokenize(preprocessed_text)
    for i in range(len(tokens)):
        print("kata: ", tokens[i])
        score = senti_indo.polarity_scores(tokens[i])
        if tokens[i] in senti_indo.lexicon:
            print("nilai lexicon: ",  senti_indo.lexicon[tokens[i]])
        else:
            print("nilai lexicon: 0")
        print("score negatif: ", score['neg'])
        print("score netral: ", score['neu'])
        print("score positif: ", score['pos'])
        print("score compound: ", score['compound'])

    score = senti_indo.polarity_scores(preprocessed_text)
    print(score)

    kata_tambahan = {
        "pudar":-5,
        "fast":1,
    }
    senti_indo.lexicon.update(kata_tambahan)

    for i in range(len(tokens)):
        print("kata: ", tokens[i])
        score = senti_indo.polarity_scores(tokens[i])
        if tokens[i] in senti_indo.lexicon:
            print("nilai lexicon: ",  senti_indo.lexicon[tokens[i]])
        else:
            print("nilai lexicon: 0")
        print("score negatif: ", score['neg'])
        print("score netral: ", score['neu'])
        print("score positif: ", score['pos'])
        print("score compound: ", score['compound'])

    score = senti_indo.polarity_scores(preprocessed_text)
    print(score)

    if score['compound'] >= 0.05 :
        print("Positive")
    elif score['compound'] <= - 0.05 :
        print("Negative")
    else :
        print("Neutral")
    
    #import data

    path = 'classification/data_preprocessing.csv'

    df = pd.read_csv(path)

    df['Ulasan_preprocessed'] = df['Ulasan_preprocessed'].fillna('')

    label_lexicon = []
    for index, row in df.iterrows():
        score = senti_indo.polarity_scores(row['Ulasan_preprocessed'])
        if score['compound'] >= 0.05 :
            label_lexicon.append(1) #positif
        elif score['compound'] <= - 0.05 :
            label_lexicon.append(2) #negatif
        else:
            label_lexicon.append(0) #netral

    # Add the preprocessed text as a new column in the DataFrame
    df['label_sentiment'] = label_lexicon
    df.to_csv('classification/data_preprocessing.csv')
    with pd.ExcelWriter(url_path, engine='xlsxwriter') as writer:
        df_ulasan.to_excel(writer, sheet_name='Ulasan', index=False),
        df_produk.to_excel(writer, sheet_name='Produk', index=False)
        df.to_excel(writer, sheet_name='Ulasan_Preprocessing', index=False)
    return 'classification/data_preprocessing.csv'