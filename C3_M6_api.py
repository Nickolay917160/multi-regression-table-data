from flask import Flask, request, jsonify
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

app = Flask(__name__)

# Глобальные переменные для хранения данных и моделей
DATA = None
TARGET = None
MODELS = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}
SELECTED_MODEL = None


@app.route('/', methods=['GET'])
def home():
    """
    Метод для корневого пути.
    Возвращает информацию о доступных эндпоинтах API.
    """
    return jsonify({
        "message": "Добро пожаловать в REST API для машинного обучения!",
        "available_endpoints": {
            "/upload": "Загрузка данных на сервер",
            "/preprocess": "Предварительная обработка данных",
            "/models": "Список доступных моделей",
            "/select_model": "Выбор модели",
            "/train": "Обучение выбранной модели",
            "/predict": "Выполнение предсказаний"
        }
    }), 200


@app.route('/upload', methods=['POST'])
def upload_data():
    """
    Метод для загрузки данных на сервер.
    Ожидает JSON с путями к файлам данных и целевых значений.
    """
    global DATA, TARGET
    try:
        data = request.json
        base_path = data.get("base_path")
        data_files = data.get("data_files")
        target_files = data.get("target_files")
        nrows = data.get("nrows", 10000)

        # Загрузка данных
        data_dfs = []
        for file in data_files:
            file_path = os.path.join(base_path, file)
            df = pd.read_csv(file_path, nrows=nrows)
            data_dfs.append(df)

        target_dfs = []
        for file in target_files:
            file_path = os.path.join(base_path, file)
            df = pd.read_csv(file_path, nrows=nrows)
            target_dfs.append(df)

        DATA = pd.concat(data_dfs, ignore_index=True)
        TARGET = pd.concat(target_dfs, ignore_index=True)

        return jsonify({"message": "Данные успешно загружены!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """
    Метод для предварительной обработки данных на сервере.
    Разделяет данные на обучающую и тестовую выборки.
    """
    global DATA, TARGET
    if DATA is None or TARGET is None:
        return jsonify({"error": "Данные не загружены!"}), 400

    try:
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            DATA, TARGET, test_size=test_size, random_state=random_state
        )

        # Сохраняем разбитые данные в глобальных переменных
        global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = X_train, X_test, y_train, y_test

        return jsonify({"message": "Данные успешно предобработаны!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/models', methods=['GET'])
def list_models():
    """
    Метод для возврата списка доступных на сервере моделей.
    """
    return jsonify({"models": list(MODELS.keys())}), 200


@app.route('/select_model', methods=['POST'])
def select_model():
    """
    Метод для выбора требуемой модели.
    """
    global SELECTED_MODEL
    model_name = request.json.get("model_name")

    if model_name not in MODELS:
        return jsonify({"error": f"Модель {model_name} недоступна!"}), 400

    SELECTED_MODEL = MODELS[model_name]
    return jsonify({"message": f"Модель {model_name} выбрана!"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Метод для выполнения предсказаний по загруженным данным.
    """
    global SELECTED_MODEL, X_TEST
    if SELECTED_MODEL is None:
        return jsonify({"error": "Модель не выбрана!"}), 400
    if X_TEST is None:
        return jsonify({"error": "Данные не загружены или не предобработаны!"}), 400

    try:
        # Выполнение предсказания
        predictions = SELECTED_MODEL.predict(X_TEST)
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/train', methods=['POST'])
def train_model():
    """
    Метод для обучения выбранной модели.
    """
    global SELECTED_MODEL, X_TRAIN, Y_TRAIN
    if SELECTED_MODEL is None:
        return jsonify({"error": "Модель не выбрана!"}), 400
    if X_TRAIN is None or Y_TRAIN is None:
        return jsonify({"error": "Данные не загружены или не предобработаны!"}), 400

    try:
        start_time = time.time()
        SELECTED_MODEL.fit(X_TRAIN, Y_TRAIN)
        training_time = time.time() - start_time

        return jsonify({"message": "Модель успешно обучена!", "training_time": training_time}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)