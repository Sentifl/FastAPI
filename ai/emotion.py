import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

model_path = "ai/emotion_analysis_model.h5"
stopword_path = "ai/stopwords.txt"
vocab_path = "ai/main_vocab.pkl"

model = load_model(model_path)

stopwords_df = pd.read_csv(stopword_path, encoding="utf-8", header=None)
stopwords = set(stopwords_df[0].tolist())

emotion_categories = ['분노', '불안', '사랑', '슬픔', '우울', '중립', '행복']

okt = Okt()

def preprocess_text(text: str):
    tokenized_sentence = okt.morphs(text)
    return [word for word in tokenized_sentence if word not in stopwords]

with open(vocab_path, 'rb') as handle:
    vocab_data = pickle.load(handle)

if isinstance(vocab_data, dict):
    tokenizer_obj = Tokenizer()
    tokenizer_obj.word_index = vocab_data
else:
    tokenizer_obj = vocab_data

maxlen = 5091 

def tokenize_and_pad(text: str):
    processed_text = preprocess_text(text)
    sequences = tokenizer_obj.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequence

def emotion_predict(sentence: str) -> dict:
    processed_sentence = tokenize_and_pad(sentence)
    
    predictions = model.predict(processed_sentence)
    
    top_2_emotions = np.argsort(predictions[0])[-2:]

    top_emotion_1 = emotion_categories[top_2_emotions[1]]
    top_emotion_2 = emotion_categories[top_2_emotions[0]]
    
    return {"emotions": [top_emotion_1, top_emotion_2]}
