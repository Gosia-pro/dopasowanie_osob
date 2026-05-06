import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Dopasowanie osób", layout="wide")



MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'
DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'
# LOAD DATA
# ======================
@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_all_participants():
    return pd.read_csv(DATA, sep=';')


# ======================
# MATCHING
# ======================

@st.cache_data
def find_similar_people(person_df, cluster_df, _model):
    features = cluster_df.drop(
        columns=["Cluster", "prediction_label", "prediction_score"], 
        errors="ignore"
    )

    combined = pd.concat([person_df, features], ignore_index=True)
    features = features[person_df.columns]
    transformed = _model.transform(combined)


    if hasattr(transformed, "torray"):
        transformed = transformed.torrray()

    user_vec = transformed[0].reshape(1, -1)
    all_vecs = transformed[1:]

    similarities = cosine_similarity(user_vec, all_vecs)[0]


    df = cluster_df.copy()
    df["similarity"] = ((similarities + 1) / 2 * 100).round(1)

    return df.sort_values("similarity", ascending=False)

st.info("📊 Analizujesz wszystkie osoby z Twojej grupy (posortowane wg podobieństwa).")

# ======================
# HELPER FUNCTIONS
# ======================
def make_histogram(df, column, title):
    fig = px.histogram(df, x=column, title=title)
    return fig

def compare_distribution(all_df, cluster_df, column):
    all_dist = all_df[column].value_counts(normalize=True)
    cluster_dist = cluster_df[column].value_counts(normalize=True)

    return pd.DataFrame({
        "Ogół": all_dist,
        "Twoja grupa": cluster_dist
    }).fillna(0)

# ======================
# SIDEBAR (INPUT)
# ======================
with st.sidebar:
    st.header("Powiedz nam coś o sobie")

    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

# ======================
# MAIN LOGIC
# ======================
model = get_model()
cluster_info = get_cluster_names()
raw_df = get_all_participants()
all_df = predict_model(model, data=raw_df)


cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
cluster_data = cluster_info.get(cluster_id, {"name": "Nieznana", "description": ""})

same_cluster_df = all_df[all_df["Cluster"] == cluster_id]

if same_cluster_df.empty:
    st.warning("Brak osób w tej grupie.")
    st.stop()

# ======================
# HEADER
# ======================
st.title("🎯 Dopasowanie osób")


st.header(f"Twoja grupa: {cluster_data['name']}")

st.header("📖 O projekcie")

st.markdown("""
Aplikacja wykorzystuje model klasteryzacji (PyCaret),
aby grupować użytkowników na podstawie ich preferencji.

Funkcjonalności:
- przypisanie do grupy
- analiza statystyczna
- dopasowanie podobnych osób (cosine similarity)
- interpretacja wyników
""")

st.markdown(cluster_data['description'])
st.metric("Liczba osób w grupie", len(same_cluster_df))

# ======================
# MATCHING
# ======================
st.subheader("👥 Najbardziej podobne osoby")

similar_people = find_similar_people(person_df, same_cluster_df, model)

# 🔥 automatyczny próg (top 20%)
quantile_threshold = similar_people["similarity"].quantile(0.8)

similar_people = similar_people[similar_people["similarity"] >= quantile_threshold]

                   
if similar_people.empty:
    st.warning("Brak osób spełniających kryterium podobieństwa.")
else:
    display_cols = [
    "age",
    "edu_level",
    "fav_animals",
    "fav_place",
    "gender",
    "similarity"
]

st.dataframe(similar_people[display_cols])

top_person = similar_people.iloc[0]
st.write(top_person)
st.success("🎯 Najbardziej podobna osoba znaleziona!")

# ======================
# COMPARISON
# ======================
st.subheader("👤 Twoje dane vs grupa")

mode_df = same_cluster_df.mode()

if not mode_df.empty:
    comparison = pd.DataFrame({
        "Ty": person_df.iloc[0],
        "Najczęstsze w grupie": mode_df.iloc[0]
    })
    st.dataframe(comparison)

# ======================
# EXPLAINABILITY
# ======================
st.subheader("🤔 Dlaczego ta grupa?")

labels = {
    "age": "wiek",
    "edu_level": "wykształcenie",
    "fav_animals": "ulubione zwierzęta",
    "fav_place": "ulubione miejsce",
    "gender": "płeć"
}

for col in person_df.columns:
    mode_series = same_cluster_df[col].mode()

    if not mode_series.empty:
        most_common = mode_series[0]

        if person_df.iloc[0][col] == most_common:
            st.write(f"✔ Masz typowe dla grupy: {labels[col]}")

# ======================
# FILTER
# ======================

view_mode = st.radio(
    "Co chcesz zobaczyć?",
    ["Cała grupa", "Najbardziej podobne osoby"]
)

if view_mode == "Cała grupa":
    filtered_df = same_cluster_df.copy()
else:
    if similar_people.empty:
        st.warning("Brak osób spełniających kryterium — pokazuję całą grupę.")
        filtered_df = same_cluster_df.copy()
    else:
        filtered_df = similar_people.copy()

selected_gender = st.selectbox("Filtruj po płci", ["Wszyscy", "Mężczyzna", "Kobieta"])

if selected_gender != "Wszyscy":
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender]


# ======================
# CHARTS
# ======================
st.divider()
st.subheader("📊 Statystyki grupy")
st.info("📊 Poniższe wykresy pokazują CAŁĄ grupę, do której zostałeś przypisany.")

fig1 = make_histogram(filtered_df, "age", "Rozkład wieku")
fig2 = make_histogram(filtered_df, "edu_level", "Rozkład wykształcenia")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)

# dodatkowe wykresy
st.plotly_chart(make_histogram(filtered_df, "fav_animals", "Ulubione zwierzęta"))
st.plotly_chart(make_histogram(filtered_df, "fav_place", "Ulubione miejsca"))
st.plotly_chart(make_histogram(filtered_df, "gender", "Płeć"))

# ======================
# COMPARISON
# ======================
st.subheader("📊 Twoja grupa vs ogół")

df_compare = compare_distribution(all_df, same_cluster_df, "fav_animals")
st.bar_chart(df_compare)

# ======================
# EDA
# ======================
if st.checkbox("🔍 Pokaż dane surowe"):
    st.dataframe(all_df)

# ======================
# SAVE
# ======================
import os
if st.button("💾 Zapisz mój profil"):
   
    file_exists = os.path.isfile("user_profile.csv")

    person_df.to_csv(
    "user_profile.csv",
    mode='a',
    header=not file_exists,
    index=False
)
st.success("Zapisano!")



# ======================
# ABOUT
# ======================
st.divider()
