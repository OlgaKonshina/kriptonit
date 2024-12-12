from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import torch

# модель для перефразирования текста
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Improved device selection:  Checks for GPU and falls back to CPU if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# функция для перефраза с помощью модели
def paraphrase(text, beams=5, grams=4):
    try:
        x = tokenizer(text, return_tensors='pt', padding=True).to(device)
        max_size = int(x.input_ids.shape[1] * 1.5 + 10)
        out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error paraphrasing '{text}': {e}")
        return text


# функция для перефраза по редковстречаемым классам эмоций
def paraphrase_data(df, emotions, columns, paraphrase_func):
    new_rows = []  # список перефразированных текстов с метками
    for index, row in df.iterrows():  # перебор по строкам
        # Это условие проверяет, имеет ли какая-либо из указанных эмоций (emotions в списке) значение 1 в текущей строке
        if any(row[emotion] == 1 for emotion in emotions):
            new_row = {'text': paraphrase_func(row['text'])}  # перефразированный текс как ключ
            for col in columns:
                new_row[col] = row[col]  # добавляем метки классов из исходного датафрейма
            new_rows.append(new_row)
    # создаем перефразированный датафрейм и соединяем с оригинальным
    df_paraphrased = pd.DataFrame(new_rows, columns=columns + ['text'])
    return pd.concat([df, df_paraphrased], ignore_index=True)


df_train = pd.read_csv('train.csv')  # Assuming 'text' column exists
emotions = ['anger', 'disgust', 'surprise', 'fear']
columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral', 'text']

df_train = paraphrase_data(df_train, emotions, columns, paraphrase)

print(df_train)
df_train.to_csv('train_and_peref.csv', index=False)
