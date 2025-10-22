import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from datetime import datetime, timedelta
import math
import google.generativeai as genai

templates = [
    "Routine maintenance check completed for {eq}. Everything nominal.",
    "Noticed a gradual spike of readings on {eq}",
    "Unexpected readings/spike from {eq} at time {td}, needs to be checked"
]
readings = ["normal","gradual_drift","spike"]
logs = []


def simulate_sensor_data(equipment, start_date='2024-01-01', months=6, freq='1h'):
    date_range = pd.date_range(start=start_date, periods=24 * 30 * months, freq=freq)
    data = pd.DataFrame({})

    for eq in equipment:
        base = np.sin(np.linspace(0, 20, len(date_range))) * 10 + 100
        noise = np.random.normal(0, 2, len(date_range))
        values = base + noise

        # Gradual drift
        drift_idx = np.random.randint(len(values) // 2, len(values))
        values[drift_idx:] += np.linspace(0, 30, len(values) - drift_idx)

        # Sudden spikes
        spike_indices = np.random.choice(len(values), size=10, replace=False)
        values[spike_indices] += np.random.randint(20, 50)

        df = pd.DataFrame({
            'timestamp': date_range,
            'equipment': eq,
            'value': values
        })
        df["has_drift"] = "normal"

        df.loc[drift_idx:, 'has_drift'] = "gradual_drift"
        df.loc[spike_indices, 'has_drift'] = "spike"

        data = pd.concat([data, df], ignore_index=True)

        percentages = ((df.has_drift.value_counts() / len(df)).values)
        print(percentages)
        for i, val in enumerate(percentages):
            for _ in range(0, math.ceil(val * 50)):
                filered_data = df.loc[np.random.choice(df.loc[df['has_drift'] == readings[i]].index)]
                logs.append({
                    "timestamp": filered_data['timestamp'],
                    "equipment": eq,
                    "log": templates[i].format(eq=eq, td=filered_data['timestamp'])
                })

    return data,logs

def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[['value']])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    return df


def correlate_logs(anomalies, logs):
    vectorizer = TfidfVectorizer(stop_words='english')
    correlated = []

    for eq in anomalies['equipment'].unique():
        eq_anoms = anomalies[(anomalies['equipment'] == eq) & (anomalies['anomaly'] == 1)]
        logs = pd.DataFrame(logs)
        eq_logs = logs[logs['equipment'] == eq]
        if eq_anoms.empty or eq_logs.empty:
            continue
        for _, a in eq_anoms.iterrows():
            a_time = a['timestamp']
            nearby_logs = eq_logs[
                (eq_logs['timestamp'] > a_time - pd.Timedelta('1 day')) &
                (eq_logs['timestamp'] < a_time + pd.Timedelta('1 day'))
                ]
            if nearby_logs.empty:
                continue
            anom_text = f"Noticed a gradual spike of readings {eq} at {a_time}"
            texts = [anom_text] + list(nearby_logs['log'])
            tfidf = vectorizer.fit_transform(texts)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
            best_idx = np.argmax(sims)
            correlated.append({
                "equipment": eq,
                "anomaly_time": a_time,
                "log": nearby_logs.iloc[best_idx]['log'],
                "similarity": sims[best_idx]
            })
    return pd.DataFrame(correlated)

def generate_summary(correlated_df):
    if correlated_df.empty:
        return "No correlated anomalies found."

    print("trying")
    summary_prompt = "Summarize the following correlated anomalies and possible causes:\n"
    for _, row in correlated_df.iterrows():
        summary_prompt += f"- Equipment: {row['equipment']}, Log: {row['log']}\n"

    genai.configure(api_key='AIzaSyCJHS356Sa7pUP6Z-sce4beOp9HBA6eym8')
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    txt = model.generate_content(summary_prompt).text
    print(txt)
    return txt


st.title("🛢️ Multi-Modal Anomaly Detection in Oil Rig Operations")

equipment = ["Pump-1", "Compressor-2", "Valve-3"]
st.sidebar.header("Simulation Settings")
months = st.sidebar.slider("months of Data", 6, 12, 15)

if st.button("Run Pipeline"):
    st.write("### Generating synthetic data...")
    df,logs = simulate_sensor_data(equipment, months=months)
    st.write("### Generating synthetic data... of length {ln}".format(ln=len(df)))

    st.write("### Detecting anomalies...")
    df_anoms = detect_anomalies(df)
    st.write(df_anoms.head())

    st.write("### Correlating anomalies with operator logs...")
    corr = correlate_logs(df_anoms, logs)
    st.write(corr.head())

    st.write("### Generating GenAI insights...")
    summary = generate_summary(corr)
    st.success(summary)

    st.write("### Visualizing Anomalies")
    fig, ax = plt.subplots()
    ax.plot(df.value, marker='+', linestyle='-', color='green')
    st.pyplot(fig)


