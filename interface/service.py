import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from joblib import load
import matplotlib.pyplot as plt

st.set_page_config(
    page_title='Предсказание поломки узлов трактора',
    page_icon=':tractor:',
    layout='wide'
)

def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)

def pred(df):
    df.replace('        -', np.nan, inplace=True)
    df.replace('-', np.nan, inplace=True)

    column_names_df = df.columns.tolist()
    for i in column_names_df:
        if (df[i].isnull().sum() / len(df[i])) * 100 > 90:
            del df[i]

    df.dropna(thresh=5, inplace=True, axis=0)

    for i in df.select_dtypes(include=['object']):
        df[i] = df[i].fillna(df[i].mode().iloc[0])

    df['Дата и время'] = pd.to_datetime(df['Дата и время'], format='%d/%m/%Y %H:%M:%S')
    df['Дата и время'] = df['Дата и время'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df['Дата и время'] = df['Дата и время'].astype('datetime64[ns]')
    
    df["cos_year_"] = make_harmonic_features(df["Дата и время"].dt.year)[0]
    df["sin_year"] = make_harmonic_features(df["Дата и время"].dt.year)[1]
    df["cos_month"] = make_harmonic_features(df["Дата и время"].dt.month)[0]
    df["sin_month"] = make_harmonic_features(df["Дата и время"].dt.month)[1]
    df["cos_day"] = make_harmonic_features(df["Дата и время"].dt.day)[0]
    df["sin_day"] = make_harmonic_features(df["Дата и время"].dt.day)[1]
    df["cos_hour_"] = make_harmonic_features(df["Дата и время"].dt.hour)[0]
    df["sin_hour"] = make_harmonic_features(df["Дата и время"].dt.hour)[1]
    df["cos_minute"] = make_harmonic_features(df["Дата и время"].dt.minute)[0]
    df["sin_minute"] = make_harmonic_features(df["Дата и время"].dt.minute)[1]
    df["cos_second"] = make_harmonic_features(df["Дата и время"].dt.second)[0]
    df["sin_second"] = make_harmonic_features(df["Дата и время"].dt.second)[1]

    df['Темп.масла двиг.,°С'] = df['Темп.масла двиг.,°С'].str.replace(',', '.').astype('float')
    df['Обор.двиг.,об/мин'] = df['Обор.двиг.,об/мин'].str.replace(',', '.').astype('float')
    df['Полож.пед.акселер.,%'] = df['Полож.пед.акселер.,%'].str.replace(',', '.').astype('float')

    return df

def to_dvs(df):
    df = pred(df)

    train_DVS = df[['Значение счетчика моточасов, час:мин','Давл.масла двиг.,кПа','Полож.пед.акселер.,%','Сост.пед.сцепл.','Темп.масла двиг.,°С','Обор.двиг.,об/мин','ДВС. Давление смазки','ДВС. Температура охлаждающей жидкости','ДВС. Частота вращения коленчатого вала','Аварийная температура охлаждающей жидкости (spn3841)','Аварийное давление масла ДВС (spn3846)','Засоренность фильтра ДВС (spn3845)','Аварийная температура масла ДВС(spn3856)','Низкий уровень ОЖ (spn3860)','Подогрев топливного фильтра (spn3865)','Скорость','Давление в пневмостистеме (spn46), кПа','Засоренность воздушного фильтра (spn3840)','Недопустимый уровень масла в гидробаке (spn3850)','Аварийная температура масла в гидросистеме (spn3849)','Выход блока управления двигателем (spn3852)','Дата и время']]
    train_DVS[['Сост.пед.сцепл.','Значение счетчика моточасов, час:мин']]=train_DVS[['Сост.пед.сцепл.','Значение счетчика моточасов, час:мин']].astype('str')
    train_DVS = train_DVS.drop(['Дата и время'],axis=1)
    train_DVS[['Давл.масла двиг.,кПа','ДВС. Давление смазки','ДВС. Температура охлаждающей жидкости']] = train_DVS[['Давл.масла двиг.,кПа','ДВС. Давление смазки','ДВС. Температура охлаждающей жидкости']].astype('float64')
    
    return train_DVS

def to_kpp(df):
    df = pred(df)
    
    train_KPP = df[['КПП. Температура масла','КПП. Давление масла в системе смазки','Нейтраль КПП (spn3843)','Засоренность фильтра КПП (spn3847)','Сост.пед.сцепл.','Давление в пневмостистеме (spn46), кПа','Недопустимый уровень масла в гидробаке (spn3850)','Аварийная температура масла в гидросистеме (spn3849)','Аварийная температура масла ГТР (spn3867)']]
    train_KPP = train_KPP.drop(['Сост.пед.сцепл.'],axis=1)
    train_KPP = train_KPP.astype('float64')

    return train_KPP

def to_tormoz(df):
    df = pred(df)
    
    train_TOR = df[['Полож.пед.акселер.,%','Нейтраль КПП (spn3843)','Аварийное давление в I контуре тормозной системы (spn3848)','Аварийное давление в II контуре тормозной системы (spn3855)']]
    train_TOR = train_TOR.astype('float64')
    return train_TOR

def to_rul(df):
    df = pred(df)
    
    train_RUL = df[['Засоренность фильтра рулевого управления (spn3844)']]
    train_RUL=train_RUL.astype('float64')

    return train_RUL

def to_el(df):
    df = pred(df)
    
    train_EL = df[['Зарядка АКБ (spn3854)', 'Отопитель (spn3853)','Электросистема. Напряжение']]
    train_EL = train_EL.astype('float64')
    return train_EL

def save_result(prediction):
    with open("prediction.pkl", "wb") as f:
        pickle.dump(prediction, f)

model_path_dvs = r"D:\tractor_forecasting\forecasting\models\model_dvs.pkl"
model_dvs = load(model_path_dvs)

model_path_kpp = r"D:\tractor_forecasting\forecasting\models\model_kpp.pkl"
model_kpp = load(model_path_kpp)

model_path_tormoz = r"D:\tractor_forecasting\forecasting\models\model_torm.pkl"
model_tormoz = load(model_path_tormoz)

model_path_rul = r"D:\tractor_forecasting\forecasting\models\model_rulev.pkl"
model_rul = load(model_path_rul)

model_path_el = r"D:\tractor_forecasting\forecasting\models\model_el.pkl"
model_el = load(model_path_el)

with st.container():

    st.sidebar.title("Параметры")
    st.title("Предсказание поломки узлов трактора")

    uploaded_file = st.sidebar.file_uploader("Загрузите ваш CSV файл")

    st.sidebar.info("Решение команды MISIS Kirovets K-700\n"
            "[Репозиторий GitHub](https://github.com/l1ghtsource/eestech-hack-tractor-forecasting).")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';')

        method = st.radio(
        "Выберите узел:",
        ["Двигатель внутреннего сгорания", "Коробка передач", "Тормозная система", 'Рулевая колонка', 'Бортовой компьютер'],
        index=0,
        )

        if method == "Коробка передач":

            df_kpp = to_kpp(df)

            prediction = model_kpp.predict_proba(df_kpp)
            df_fin = pd.DataFrame({
                'normal': prediction[:,0],
                'anomaly': prediction[:,1],
                'problem' : prediction[:,2]
            })
            df_fin['normal'] = df_fin['normal'].mean()
            df_fin['anomaly'] = df_fin['anomaly'].mean()
            df_fin['problem'] = df_fin['problem'].mean()

            df_fin = df_fin.drop_duplicates()

            st.write("## Результат")
            st.metric("Вероятность поломки коробки передач", round(df_fin['problem'], 5))
            st.metric("Вероятность аномалии в телеметрии коробки передач", round(df_fin['anomaly'], 5))
            st.metric("Вероятность нормального состояния коробки передач", round(df_fin['normal'], 5))

            st.bar_chart(df_fin, y=['normal', 'anomaly', 'problem'])

            st.write('Вы можете найти необходимые запчасти по ссылке: https://kirovets-ptz.com/zapasnye-chasti/')

            if st.button("Сохранить результат", key="save_prediction"):
                save_result(prediction)
                st.success("Результат успешно сохранен!")
        
        elif method == "Двигатель внутреннего сгорания":

            df_dvs = to_dvs(df)

            prediction = model_dvs.predict_proba(df_dvs)
            df_fin = pd.DataFrame({
                'normal': prediction[:,0],
                'anomaly': prediction[:,1],
                'problem' : prediction[:,2]
            })
            df_fin['normal'] = df_fin['normal'].mean()
            df_fin['anomaly'] = df_fin['anomaly'].mean()
            df_fin['problem'] = df_fin['problem'].mean()

            df_fin = df_fin.drop_duplicates()

            st.write("## Результат")
            st.metric("Вероятность поломки ДВС", round(df_fin['problem'], 5))
            st.metric("Вероятность аномалии в телеметрии ДВС", round(df_fin['anomaly'], 5))
            st.metric("Вероятность нормального состояния ДВС", round(df_fin['normal'], 5))

            st.bar_chart(df_fin, y=['normal', 'anomaly', 'problem'])

            st.write('Вы можете найти необходимые запчасти по ссылке: https://kirovets-ptz.com/zapasnye-chasti/')

            if st.button("Сохранить результат", key="save_prediction"):
                save_result(prediction)
                st.success("Результат успешно сохранен!")
        
        elif method == "Тормозная система":

            df_tormoz = to_tormoz(df)

            prediction = model_tormoz.predict_proba(df_tormoz)
            df_fin = pd.DataFrame({
                'normal': prediction[:,0],
                'anomaly': prediction[:,1],
                'problem' : prediction[:,2]
            })
            df_fin['normal'] = df_fin['normal'].mean()
            df_fin['anomaly'] = df_fin['anomaly'].mean()
            df_fin['problem'] = df_fin['problem'].mean()

            df_fin = df_fin.drop_duplicates()

            st.write("## Результат")
            st.metric("Вероятность поломки тормозной системы", round(df_fin['problem'], 5))
            st.metric("Вероятность аномалии в телеметрии тормозной системы", round(df_fin['anomaly'], 5))
            st.metric("Вероятность нормального состояния тормозной системы", round(df_fin['normal'], 5))

            st.bar_chart(df_fin, y=['normal', 'anomaly', 'problem'])

            st.write('Вы можете найти необходимые запчасти по ссылке: https://kirovets-ptz.com/zapasnye-chasti/')

            if st.button("Сохранить результат", key="save_prediction"):
                save_result(prediction)
                st.success("Результат успешно сохранен!")

        elif method == "Рулевая колонка":

            df_rul = to_rul(df)

            prediction = model_rul.predict_proba(df_rul)
            df_fin = pd.DataFrame({
                'normal': prediction[:,0],
                'anomaly': prediction[:,1],
                'problem' : prediction[:,2]
            })
            df_fin['normal'] = df_fin['normal'].mean()
            df_fin['anomaly'] = df_fin['anomaly'].mean()
            df_fin['problem'] = df_fin['problem'].mean()

            df_fin = df_fin.drop_duplicates()

            st.write("## Результат")
            st.metric("Вероятность поломки рулевой колонки", round(df_fin['problem'], 5))
            st.metric("Вероятность аномалии в телеметрии рулевой колонки", round(df_fin['anomaly'], 5))
            st.metric("Вероятность нормального состояния рулевой колонки", round(df_fin['normal'], 5))

            st.bar_chart(df_fin, y=['normal', 'anomaly', 'problem'])

            st.write('Вы можете найти необходимые запчасти по ссылке: https://kirovets-ptz.com/zapasnye-chasti/')

            if st.button("Сохранить результат", key="save_prediction"):
                save_result(prediction)
                st.success("Результат успешно сохранен!")

        elif method == "Бортовой компьютер":

            df_el = to_el(df)

            prediction = model_el.predict_proba(df_el)
            df_fin = pd.DataFrame({
                'normal': prediction[:,0],
                'anomaly': prediction[:,1],
                'problem' : prediction[:,2]
            })
            df_fin['normal'] = df_fin['normal'].mean()
            df_fin['anomaly'] = df_fin['anomaly'].mean()
            df_fin['problem'] = df_fin['problem'].mean()

            df_fin = df_fin.drop_duplicates()

            st.write("## Результат")
            st.metric("Вероятность поломки бортового компьютера", round(df_fin['problem'], 5))
            st.metric("Вероятность аномалии в телеметрии бортового компьютера", round(df_fin['anomaly'], 5))
            st.metric("Вероятность нормального состояния бортового компьютера", round(df_fin['normal'], 5))

            st.bar_chart(df_fin, y=['normal', 'anomaly', 'problem'])

            st.write('Вы можете найти необходимые запчасти по ссылке: https://kirovets-ptz.com/zapasnye-chasti/')

            if st.button("Сохранить результат", key="save_prediction"):
                save_result(prediction)
                st.success("Результат успешно сохранен!")

st.sidebar.markdown("---")