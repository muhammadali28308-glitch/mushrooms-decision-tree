import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Page ki setting
st.set_page_config(page_title="Mushroom Predictor", page_icon="🍄")

@st.cache_resource
def train_model():
    df = pd.read_csv("mushrooms.csv")
    X = df.drop("type", axis=1)
    y = df["type"]

    categorical_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X, y)
    
    # Baaki columns ke liye default values (mode) nikaal lo
    default_values = df.mode().iloc[0].to_dict()
    
    return pipeline, X.columns.tolist(), default_values

model, all_columns, default_values = train_model()

st.title("🍄 Mushroom Edibility Predictor")
st.write("Neeche di gayi details daalein taake pata chale ke mushroom khane layak hai ya zehrila.")

odor = st.selectbox("Odor (Badboo)", ['a', 'l', 'c', 'f', 'm', 'n', 'p', 's', 'y'])
cap_shape = st.selectbox("Cap Shape (Top ki shakal)", ['b', 'c', 'x', 'f', 'k', 's'])
cap_color = st.selectbox("Cap Color (Top ka rang)", ['n', 'y', 'w', 'g', 'e', 'p', 'b', 'u', 'c', 'r'])
population = st.selectbox("Population (Aas paas ki aabadi)", ['a', 'n', 's', 'v', 'y', 'c'])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'odor': [odor],
        'cap_shape': [cap_shape],
        'cap_color': [cap_color],
        'population': [population]
    })
    
    # Baaki columns ke liye asli default values daalo (x ki jagah)
    for col in all_columns:
        if col not in input_data.columns:
            input_data[col] = default_values[col]
    
    prediction = model.predict(input_data)[0]
    
    # Debug ke liye (taake pata chale model kya predict kar raha hai)
    st.write(f"🔍 *Debug: Model ne predict kiya '{prediction}'*")
    
    # Condition ko thora robust kiya hai (strings handle karne ke liye)
    if str(prediction).lower() == 'p':
        st.error("⚠️ Warning: Ye mushroom ZEHRILA (Poisonous) hai! Isay mat khao.")
    else:
        st.success("✅ Good News: Ye mushroom KHANE LAYAK (Edible) hai!")