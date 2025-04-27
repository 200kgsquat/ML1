import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Загрузка модели
model = joblib.load(r'E:\Test_intern\best_model.pkl')

st.title("🎯 Прогноз сдачи экзамена")
st.markdown("Загрузите CSV файл с вашими данными:")

# Функция препроцессинга
def preprocess_data(df):
    df_processed = df.copy()

    # Маппинг бинарных колонок
    df_processed['Сон накануне'] = df_processed['Сон накануне'].map({'Нет': 0, 'Да': 1})
    df_processed['Энергетиков накануне'] = df_processed['Энергетиков накануне'].map({'0': 0, '1': 1, '2-3': 2, '4+': 3})

    # OrdinalEncoder для категорий
    encoder = OrdinalEncoder()
    encoded_cols = ['Настроение', 'Посещаемость занятий', 'Время подготовки']

    df_processed[encoded_cols] = encoder.fit_transform(df_processed[encoded_cols])

    return df_processed

# Загрузка CSV файла
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file is not None:
    try:
        # Чтение файла
        data = pd.read_csv(uploaded_file)

        required_columns = [
            'Контрольная 1', 'Контрольная 2', 'Контрольная 3', 
            'Сон накануне', 'Настроение', 'Энергетиков накануне', 
            'Посещаемость занятий', 'Время подготовки'
        ]

        if all(col in data.columns for col in required_columns):
            st.success("Файл успешно загружен!")

            # Показываем исходные данные
            st.subheader("📝 Ваши загруженные данные:")
            st.dataframe(data[required_columns])

            # Препроцессинг данных
            X = preprocess_data(data[required_columns])

            # Выбор строки
            selected_row = st.number_input(
                "Выберите номер студента для прогноза (начинается с 0):", 
                min_value=0, 
                max_value=len(X) - 1, 
                value=0
            )

            row = X.iloc[[selected_row]]  # Двойные скобки чтобы получить DataFrame

            # Предсказание
            prediction = model.predict(row)[0]
            probability = model.predict_proba(row)[0][1]

            # Вывод красивого результата
            st.subheader(" Результат прогноза:")
            if prediction == 1:
                st.success(f"✅ Студент **СДАЛ** экзамен с вероятностью {probability:.2%}")
            else:
                st.error(f"❌ Студент **НЕ СДАЛ** экзамен с вероятностью {1 - probability:.2%}")

        else:
            st.error(f"Ошибка: в файле должны быть колонки: {required_columns}")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")

else:
    st.info("Пожалуйста, загрузите CSV файл для прогноза.")
