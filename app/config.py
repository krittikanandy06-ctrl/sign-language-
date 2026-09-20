import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "sign_language_rec_app_secret_key_2026"
    JSON_SORT_KEYS = False
