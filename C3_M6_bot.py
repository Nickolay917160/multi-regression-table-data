import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# URL вашего Flask-сервера
SERVER_URL = "http://127.0.0.1:5000"

# Функция для отправки данных на сервер
async def upload_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Пример данных для загрузки
        data = {
            "base_path": "/home/c3/Загрузки/BIG DATA",
            "data_files": ["df_0.csv", "df_1.csv"],
            "target_files": ["target_0.csv", "target_1.csv"],
            "nrows": 10000
        }
        response = requests.post(f"{SERVER_URL}/upload", json=data)
        await update.message.reply_text(response.json().get("message", "Ошибка при загрузке данных"))
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Функция для предварительной обработки данных
async def preprocess_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        data = {"test_size": 0.2}
        response = requests.post(f"{SERVER_URL}/preprocess", json=data)
        await update.message.reply_text(response.json().get("message", "Ошибка при предобработке данных"))
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Функция для получения списка доступных моделей
async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = requests.get(f"{SERVER_URL}/models")
        models = response.json().get("models", [])
        if models:
            await update.message.reply_text(f"Доступные модели: {', '.join(models)}")
        else:
            await update.message.reply_text("Модели не найдены.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Функция для выбора модели
async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model_name = context.args[0] if context.args else None
        if not model_name:
            await update.message.reply_text("Укажите имя модели. Например: /select_model RandomForest")
            return

        data = {"model_name": model_name}
        response = requests.post(f"{SERVER_URL}/select_model", json=data)
        await update.message.reply_text(response.json().get("message", "Ошибка при выборе модели"))
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Функция для обучения модели
async def train_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = requests.post(f"{SERVER_URL}/train")
        message = response.json().get("message", "Ошибка при обучении модели")
        training_time = response.json().get("training_time", "Время неизвестно")
        await update.message.reply_text(f"{message}\nВремя обучения: {training_time:.2f} секунд")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Функция для выполнения предсказаний
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = requests.post(f"{SERVER_URL}/predict")
        predictions = response.json().get("predictions", [])
        if predictions:
            await update.message.reply_text(f"Предсказания: {predictions}")
        else:
            await update.message.reply_text("Предсказания не получены.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Основная функция для запуска бота
async def main():
    # Вставьте ваш токен Telegram-бота
    TOKEN = "7549616008:AAGicSrGxpu6DZYxU2acX5JCAx5VFlP7Jqo"

    # Создаем приложение
    application = Application.builder().token(TOKEN).build()

    # Регистрация команд
    application.add_handler(CommandHandler("start", lambda update, context: update.message.reply_text(
        "Добро пожаловать! Используйте команды:\n"
        "/upload - Загрузить данные\n"
        "/preprocess - Предобработать данные\n"
        "/models - Список доступных моделей\n"
        "/select_model <имя> - Выбрать модель\n"
        "/train - Обучить модель\n"
        "/predict - Получить предсказания"
    )))
    application.add_handler(CommandHandler("upload", upload_data))
    application.add_handler(CommandHandler("preprocess", preprocess_data))
    application.add_handler(CommandHandler("models", list_models))
    application.add_handler(CommandHandler("select_model", select_model))
    application.add_handler(CommandHandler("train", train_model))
    application.add_handler(CommandHandler("predict", predict))

    # Запуск бота
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    print("Бот запущен...")
    await application.updater.stop()
    await application.stop()
    await application.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())