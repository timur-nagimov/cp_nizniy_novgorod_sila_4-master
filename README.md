## Создаем виртуальное окружение и устанавливаем библиотеки:
Команды предполагают использование MacOS/Ubuntu
* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`
## Поднимаем базу данных:
* Переходим `cd backend`
* Через `sudo docker-compose up` поднимается контейнер с postgres
* Остальные операции выполняем в отдельном терминале(тоже через виртуальное окружение)
## Создаем и применяем миграции
* Применяем миграции из backend папки: `alembic upgrade head`
## Поднимаем модель:
* Из главной папки подгружаем в БД данные: python3 upload_to_bd.py
* Находясь в главной папке, прописываем: `streamlit run ml/streamlit_app.py`

## Описание ML части
* splitter.ipynb - нарезка исходного документа на разделы и подразделы
* ml/rag.py - функции и классы для построения RAG пайплайна на llama-index
* ml/prompts.py - промпты используемые в пайплайне
* ml/preset.py - функции для поиска похожих запросов в имеющейся базе знаний
* ml/streamlit_app.py - создание интерфейса на streamlit 
* ml/custom_response.py - измененный класс CustomQueryComponent для форматирования выдающейся в пайплайне информации
* ml/retriever.py - функции для загрузки индекса и ретривера
* ml/yandex_gpt.py - функции для инициализации YandexGPT 

## Метрики  
С помощью фреймворка [RAGAS](ragas.io) сгенерированы вопросы и ответы по чанкам (разделам документа), доступные в файле ragas_df.pkl.  
Получены следующие метрики - [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) 78.4, [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) 93.2, [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) 80.7, [Response Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) 85.1
