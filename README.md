# Проектный практикум III.(Криптонит). Команда 22.
### Члены команды:
* Дмитрий Шабанов - Leader
* Коньшина Ольга - Data scientist
* Прохорова Екатарина - Data scientist
* Ильиных Виктория - ML engineer
* Татьяна Егоренкова - Data analyst
* Василий Воробьев - ML engineer

### Задача
Обучить языковую модель для классификации эмоций в текстах на русском языке. Количество эмоций - 7, при этом текст может иметь не одну эмоцию, а несколько.
###  Этапы выполнения проекта.
#### Разведочный анализ данных.
Особенности выявленные в предоставленных данных:
* В данных  в данных присутствует дисбаланс классов. Явное преобладание класса 'joy'. Недостаточно представлены классы 'disgust', 'fear'.
* Значительная часть данных — результат машинного перевода с английского языка, что отражается на стиле изложения и лексике.
* Присутствует значимая неточность в разметке, когда представленный класс явно не соответствует контексту.
* Значительная часть текстовых данных содержит ошибки/опечатки,
* Отсутствует/нарушена пунктуация

#### Добавление данных в обучающий набор.
* Добавление данных с помощью модели rut5-base-paraphraser. Идея заключается в том, чтобы добавить тексты в мало представленные классы с помощью перефразирования с сохранением меток.
  Скрипт - adding data/perephras.py
* Добавление данных из датасета CEDR. CEDR - Корпус для выявления эмоций в предложениях русскоязычных текстов из разных социальных источников содержит 9410 комментариев, помеченных по 5 категориям эмоций (радость, грусть, удивление, страх и гнев). Скрипт - adding data/perephras.py
* Добавление данных из датасета 'Djacon/ru-izard-emotions'. Он содержит 30 000 комментариев с Reddit, размеченных по 10 категориям эмоций (joy, sadness, anger, enthusiasm, surprise, disgust, fear, guilt, shame and neutral). Наборы данных были переведены с помощью точного переводчика DeepL и дополнительно обработаны. Добавление RuIzardEmotions помогло нам улушить работу модели.
  
#### Предобработка данных.
* Выполнены стандартные этапы очистки данных (удаление знаков препинания, удаление emoji, сторонних символов, приведение к нижнему регистру).
* Выполнено удаление стоп слов, лемматизации. Эти этапы обраброки снижают качество обчения модели.
* Выполнено удаление редковстречаемых слов. Скрипт - data preparation/clean_data.ipynb
* Выполнена балансировка классов методами oversamplinng и undersanpling - получили ухудшение метрик.
* Выполнена лемматизация разными способами: используя библиотеку nltk и pymystem3.
  
#### Обучение модели.
Было рассмотрено несколько потенциальных вариантов архитектуры:
* Было обучено несколько моделей, основанных на архитектуре BERT(ruBert-base, rudert-tyny2,rubert-base-emotion-russian-cedr-m7), с различными наборами гиперпараметров. Наилучшие результаты были получены с использованием модели ruBert-base и оптимизатора Adam.(F1 weighted на тестовых данных 0.59),дообученной на датасете с добавлением данных RuIzardEmotions. Использование сложной предварительной обработки ухудшают качество модели.
* Рассматривался вариант обучения нескольких отдельных логистических регрессий под предсказания вероятности отнесения текста к каждому классу. Возможно потребуется решить задачу установки минимальных пороговых ограничений по привязке конкретного класса к тексту. Данный подход не дал улучшения результата. 
* Выполнено обучения и дополнительными слоями нейроной в архитектуре неросети, также дообучались веса предобучееной сети. 
* Меняли фунции потерь, learning rate, threshold, batch size, weight_decay, число эпох.
* К видемым улучшениям привело только понижение threshold
* Была тестирована модель 'MilaNLProc/xlm-emo-t',которая является усовершенствованной версией модели XLM-T.XLM-T — модель для обучения и оценки многоязычных языковых моделей в Twitter.
* Поизведен обширный анализ разных моделей глубокого обучения: LaBSE, DistilBERT, RoBERTa, XLNet, ALBERT, RuBERT и RuGPT-3. Анализ и результаты содержатся в ноутбуке Model analysis.ipynb.

### Лучший результат получен при обучении модели ruBert-base с использванием расширенного датасета и понижением threshold .
(ноутбук с обучением - Emotion_expand/Emotion_расширенный.ipynb) Презентация - presentation/Project v.4.pdf
Score = 0,59505
