from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the model and tokenizer
model = load_model("../models/lstm_text_generation_model.keras")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to generate new text
def generate_text(seed_text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=29, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

if __name__ == "__main__":
    seed = "you call me all"
    generated = generate_text(seed, 5)
    print(generated)
