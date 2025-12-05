import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(page_title="Car Price Prediction",
                   page_icon="🚗", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "Ridge_model.pkl"


@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


def strip_mileage(s):
    try:
        s = s.strip(' kmpl').strip(' km/kg')
    except:
        s = float(s)
    return float(s)


def strip_engine(s):
    try:
        s = s.strip(' CC')
    except:
        s = float(s)
    return float(s)


def strip_power(s):
    try:
        s = s.strip(' bhp')
        return float(s)
    except:
        if s == '':
            return np.nan
        else:
            return float(s)


def prepare_features(df):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    df_proc['mileage'] = df_proc['mileage'].apply(lambda x: strip_mileage(x))
    df_proc['mileage'].fillna(df_proc['mileage'].median(), inplace=True)
    df_proc['engine'] = df_proc['engine'].apply(lambda x: strip_engine(x))
    df_proc['engine'].fillna(df_proc['engine'].median(), inplace=True)
    df_proc['max_power'] = df_proc['max_power'].apply(lambda x: strip_power(x))
    df_proc['max_power'].fillna(df_proc['max_power'].median(), inplace=True)
    df_proc['seats'].fillna(df_proc['seats'].median(), inplace=True)
    df_proc.drop_duplicates(subset=df_proc.columns.drop(
        'selling_price'), inplace=True, keep='first')
    df_proc.reset_index(inplace=True)
    df_proc.drop('index', axis=1, inplace=True)
    df_proc.drop('torque', axis=1, inplace=True)
    df_proc['seats'] = df_proc['seats'].astype(int)
    df_proc['engine'] = df_proc['engine'].astype(int)
    df_proc['name'] = df_proc['name'].apply(lambda x: x.split()[0])

    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    encoder.fit(df_proc['name'].values.reshape(-1, 1))
    encoded_name = pd.DataFrame(encoder.transform(
        df_proc['name'].values.reshape(-1, 1)).toarray(), columns=encoder.get_feature_names_out())
    df_proc_enc = pd.concat(
        [df_proc.drop('name', axis=1), encoded_name], axis=1).reset_index(drop=True)

    encoder2 = OneHotEncoder(drop='first', handle_unknown='ignore')
    encoder2.fit(df_proc[['fuel', 'seller_type', 'transmission', 'owner']])
    encoded_cat = pd.DataFrame(encoder2.transform(
        df_proc[['fuel', 'seller_type', 'transmission', 'owner']]).toarray(), columns=encoder2.get_feature_names_out())
    df_proc_enc = pd.concat([df_proc_enc.drop(
        ['fuel', 'seller_type', 'transmission', 'owner'], axis=1), encoded_cat], axis=1).reset_index(drop=True)
    X = df_proc_enc.drop('selling_price', axis=1)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    y = df_proc_enc['selling_price']

    return X, y


# Загружаем модель
try:
    MODEL = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# --- Основной интерфейс ---
st.title("🚗 Car Price Prediction App")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

try:
    X, y = prepare_features(df)
    predictions = MODEL.predict(X)

except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

# Создание вкладок
tab1, tab2, tab3 = st.tabs([
    "📈 EDA Dashboard",
    "🔮 Make Prediction",
    "📊 Model Analysis"
])

with tab1:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("📋 Basic Information")
        st.write(f"**Training samples:** {df.shape[0]}")
        st.write(f"**Features:** {df.shape[1]}")

        # Пропущенные значения
        missing_values = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Feature': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': (missing_values.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]

        if not missing_df.empty:
            st.subheader("⚠️ Missing Values")
            st.dataframe(missing_df, use_container_width=True)

    with col2:
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # Визуализации
    st.subheader("📈 Visualizations")

    viz_option = st.selectbox(
        "Select Visualization",
        ["Price Distribution", "Price vs Year", "Price vs Mileage",
         "Fuel Type Distribution", "Transmission Type", "Correlation Heatmap"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    if viz_option == "Price Distribution":
        ax.hist(df['selling_price'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Car Prices')
        ax.ticklabel_format(style='plain', axis='x')

    elif viz_option == "Price vs Year":
        yearly_avg = df.groupby('year')['selling_price'].mean().reset_index()
        ax.plot(yearly_avg['year'],
                yearly_avg['selling_price'], marker='o', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Price')
        ax.set_title('Average Price by Year')
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, alpha=0.3)

    elif viz_option == "Price vs Mileage":
        ax.scatter(df['km_driven'], df['selling_price'],
                   alpha=0.5, s=20)
        ax.set_xlabel('Kilometers Driven')
        ax.set_ylabel('Price')
        ax.set_title('Price vs Kilometers Driven')
        ax.ticklabel_format(style='plain', axis='both')

    elif viz_option == "Fuel Type Distribution":
        fuel_counts = df['fuel'].value_counts()
        ax.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette("husl", len(fuel_counts)))
        ax.set_title('Distribution by Fuel Type')

    elif viz_option == "Transmission Type":
        trans_counts = df['transmission'].value_counts()
        bars = ax.bar(trans_counts.index, trans_counts.values,
                      color=sns.color_palette("husl", len(trans_counts)))
        ax.set_xlabel('Transmission Type')
        ax.set_ylabel('Count')
        ax.set_title('Distribution by Transmission Type')
        ax.bar_label(bars)

    elif viz_option == "Correlation Heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax)
        ax.set_title('Feature Correlation Heatmap')

    st.pyplot(fig)


# Вкладка 3: Предсказание
with tab2:
    st.header("Make Price Prediction")

    if MODEL is None:
        st.warning("⚠️ Error load model")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚗 Car Specifications")

            year = st.slider("Year", 2000, 2023, 2015)
            km_driven = st.number_input(
                "Kilometers Driven", 0, 500000, 50000, step=1000)

            fuel = st.selectbox(
                "Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            seller_type = st.selectbox(
                "Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
            transmission = st.selectbox(
                "Transmission", ["Manual", "Automatic"])
            owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner",
                                           "Fourth & Above Owner", "Test Drive Car"])

        with col2:
            st.subheader("🔧 Technical Specifications")

            mileage = st.number_input(
                "Mileage (kmpl)", 5.0, 40.0, 20.0, step=0.1)
            engine = st.number_input("Engine (CC)", 500, 5000, 1500, step=100)
            max_power = st.number_input(
                "Max Power (bhp)", 30, 500, 100, step=5)
            seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9])

            if st.button("💰 Predict Price", type="primary"):
                input_data = {
                    'year': year,
                    'km_driven': km_driven,
                    'fuel': fuel,
                    'seller_type': seller_type,
                    'transmission': transmission,
                    'owner': owner,
                    'mileage': mileage,
                    'engine': engine,
                    'max_power': max_power,
                    'seats': seats
                }

                prediction = MODEL.predict(input_data)

                if prediction is not None:
                    st.success(f"### Predicted Price: ₹{prediction:,.2f}")

                    # Дополнительная информация
                    st.info("💡 **Price Range Interpretation:**")

                    if prediction < 500000:
                        st.write("**Category:** Budget Car")
                        st.write(
                            "**Typical examples:** Maruti, Hyundai entry-level models")
                    elif prediction < 1500000:
                        st.write("**Category:** Mid-Range Car")
                        st.write(
                            "**Typical examples:** Honda, Toyota, VW mid-size models")
                    elif prediction < 4000000:
                        st.write("**Category:** Premium Car")
                        st.write(
                            "**Typical examples:** BMW, Mercedes, Audi entry-level")
                    else:
                        st.write("**Category:** Luxury Car")
                        st.write(
                            "**Typical examples:** High-end BMW, Mercedes, luxury brands")


# Вкладка 3: Анализ модели
with tab3:
    st.header("Model Analysis")

    if MODEL is None:
        st.warning("⚠️ Please load a model first")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Model Information")

            model_name = MODEL.__class__.__name__
            st.write(f"**Model Type:** {model_name}")

            # Коэффициенты для линейных моделей
            if hasattr(MODEL, 'coef_'):
                st.subheader("📈 Model Coefficients")

                coef_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Coefficient': MODEL.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)

                st.dataframe(coef_df, use_container_width=True)

        with col2:
            st.subheader("🎯 Model Performance Insights")

            st.write("""
                **Interpretation Guide:**
                
                - **R² Score:** Closer to 1 is better (explained variance)
                - **MSE:** Lower is better (prediction error)
                - **Feature Importance:** Which factors most affect price
                - **Residuals:** Should be randomly distributed around zero
                """)
