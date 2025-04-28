import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

# Определяем относительный путь к модели через pathlib
script_path = Path(__file__).resolve()
model_path = script_path.parent / 'best_model.pkl'

# Пытаемся загрузить модель
try:
    model = joblib.load(model_path)
except ModuleNotFoundError as e:
    st.error(f"❌ Не удалось загрузить модель: отсутствует модуль '{e.name}'. Добавьте его в requirements.txt и перезапустите деплой.")
    st.stop()
except FileNotFoundError:
    st.error(f"❌ Файл модели не найден: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"❌ Произошла ошибка при загрузке модели: {e}")
    st.stop()

# Интерфейс Streamlit
st.title("🎯 Прогноз сдачи экзамена")
st.markdown("Загрузите CSV файл с вашими данными:")

# Функция препроцессинга
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    df_processed['Сон накануне'] = df_processed['Сон накануне'].map({'Нет': 0, 'Да': 1})
    df_processed['Энергетиков накануне'] = df_processed['Энергетиков накануне'].map({'0': 0, '1': 1, '2-3': 2, '4+': 3})

    encoder = OrdinalEncoder()
    encoded_cols = ['Настроение', 'Посещаемость занятий', 'Время подготовки']
    df_processed[encoded_cols] = encoder.fit_transform(df_processed[encoded_cols])
    return df_processed

# Загрузка CSV файла
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        required_columns = [
            'Контрольная 1', 'Контрольная 2', 'Контрольная 3',
            'Сон накануне', 'Настроение', 'Энергетиков накануне',
            'Посещаемость занятий', 'Время подготовки'
        ]
        if not all(col in data.columns for col in required_columns):
            st.error(f"Ошибка: в файле должны быть колонки: {required_columns}")
        else:
            st.success("Файл успешно загружен!")
            st.subheader("📝 Ваши загруженные данные:")
            st.dataframe(data[required_columns], height=200)

            X = preprocess_data(data[required_columns])

            # 🔍 Индивидуальный прогноз
            st.subheader("🔍 Индивидуальный прогноз")
            selected_row = st.number_input(
                f"Выберите номер студента (от 0 до {len(X)-1}):", 0, len(X)-1, 0
            )
            row = X.iloc[[selected_row]]
            pred_single = model.predict(row)[0]
            prob_single = model.predict_proba(row)[0][1]
            if pred_single == 1:
                st.success(f"✅ Студент {selected_row} **СДАЛ** экзамен с вероятностью {prob_single:.2%}")
            else:
                st.error(f"❌ Студент {selected_row} **НЕ СДАЛ** экзамен с вероятностью {(1 - prob_single):.2%}")

            # 📊 Прогноз для всех студентов с вероятностями
            st.subheader("📊 Прогноз для всех студентов")
            if st.button("Показать прогнозы для всех студентов"):
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]
                results = pd.DataFrame({
                    'Студент': list(range(len(X))),
                    'Прогноз': ['СДАЛ' if p == 1 else 'НЕ СДАЛ' for p in preds],
                    'Вероятность': [
                        f"{probs[i]:.2%}" if preds[i] == 1 else f"{(1 - probs[i]):.2%}"
                        for i in range(len(probs))
                    ]
                })

                st.dataframe(results, height=300)

                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать результаты",
                    data=csv,
                    file_name="all_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
else:
    st.info("Пожалуйста, загрузите CSV файл для прогноза.")
