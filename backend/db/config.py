import os
from os import environ

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

load_dotenv(dotenv_path)

DATABASE_HOST: str = environ.get("DATABASE_HOST")
DATABASE_USERNAME: str = environ.get("DATABASE_USERNAME")
DATABASE_PASSWORD: str = environ.get("DATABASE_PASSWORD")
DATABASE_PORT: str = str(environ.get("DATABASE_PORT"))
DATABASE_NAME: str = environ.get("DATABASE_NAME")
