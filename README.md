## 🧠 Approach

- Generated **synthetic data** and added noise with a frequency of **1 hour**.  
- Introduced **sudden and gradual spikes** in selected points by applying weighted deviations.  
- Applied **Isolation Forest** to identify anomalies.  
- Identified **nearby logs (within 1 day)** and calculated the **most similar log text** based on text similarity (correlated logs).  
- Generated a **summary** from the correlated logs.

---

## ⚙️ Assumptions

- Synthetic data generated at **1-hour frequency**.  
- Used a **sine wave pattern** for smooth baseline data generation.  
- Considered **1-day timespan** to search for correlated logs.

---

## ⚠️ Failure Points

- **Incorrect log-to-anomaly mapping** may occur.  
- **TF-IDF** may fail to capture **contextual meaning**.  
- Potential **false positives / false negatives** from the Isolation Forest model.  
- Logs generated are **not meaningful**, reducing correlation quality.

---

## 🚀 Future Work

- Improve **text embeddings** for better contextual understanding.  
- Enhance **log generation** to produce more realistic and informative entries.  
- **Evaluate metrics** against ground truth for quantitative assessment.  
- Experiment with **different anomaly detection techniques**.  
- **Visualize** predicted vs actual anomalies for better interpretability.
