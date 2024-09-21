import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from konlpy.tag import Okt

model_path = "ai/emotion_analysis_model.h5"
vocab_path = "ai/main_vocab.pkl"
stopword_path = "ai/stopwords.txt"

model = load_model(model_path)

with open(vocab_path, 'rb') as f:
    main_vocab = pickle.load(f)

stopwords_df = pd.read_csv(stopword_path, encoding="utf-8", header=None)
stopwords = stopwords_df[0].tolist() 

tokenizer = Okt()

def preprocess(sentence: str) -> np.ndarray:

    tokenized_sentence = tokenizer.morphs(sentence)

    result = [main_vocab[word] for word in tokenized_sentence if word not in stopwords and word in main_vocab]
    
    one_hot_encoding = np.zeros(5091)
    one_hot_encoding[result] = 1
    
    return np.array([one_hot_encoding]) 

def emotion_predict(sentence: str) -> str:
    processed_sentence = preprocess(sentence)
    prediction = model.predict(processed_sentence)

    label_dict = {0: '분노', 1: '불안', 2: '사랑', 3: '슬픔', 4: '우울', 5: '중립', 6: '행복'}
    predicted_label = label_dict[np.argmax(prediction)]
    
    return predicted_label
