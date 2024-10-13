## Создаем виртуальное окружение и устанавливаем библиотеки:
Прописываем: `export PYTHONPATH=$PYTHONPATH:/home/user1/hack_dev/backend` - с особенностями пути после скачивания репозитория

Команды предполагают использование MacOS/Ubuntu
* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`
## Поднимаем базу данных:
* Переходим `cd backend`
* Через `sudo docker-compose up` поднимается контейнер с postgres
* Остальные операции выполняем в отдельном терминале(тоже через виртуальное окружение)
## Создаем и применяем миграции
* Находясь в папке `/backend`, прописываем `alembic revision --autogenerate -m 'init'`
* Применяем миграции: `alembic upgrade head`
## Поднимаем модель:
* Из главной папки подгружаем в БД данные: python3 upload_to_bd.py
* Находясь в главной папке, прописываем: `streamlit run ml/streamlit_app.py`

## Описание ML части
* NPA_splitter.ipynb - нарезка исходного документа на разделы и подразделы
* ml/rag.py - функции и классы для построения RAG пайплайна на llama-index, в качестве LLM - Qwen 2.5 7B Instruct
* ml/prompts.py - промпты используемые в пайплайне
* ml/preset.py - функции для поиска похожих запросов в имеющейся базе знаний
* ml/streamlit_app.py - создание интерфейса на streamlit 
