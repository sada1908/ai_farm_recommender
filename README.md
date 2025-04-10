# 🌾 Resilient Crop Recommendation AI

**Resilient Crop AI** is an agentic, cross-dataset crop recommendation system designed to align farm-specific sustainability conditions with market profitability using fuzzy logic, spectral clustering, and probabilistic distribution reasoning. It is built to empower farmers and agronomists with data-backed, explainable, and adaptive crop planning tools.

---

## 🔍 Problem Statement

Farmers face fragmented data: agronomic conditions and sustainability metrics are stored separately from real-time market trends. Existing advisory systems often fail to bridge this gap, making decisions either profit-centric or blindly yield-based. There is a critical need for an AI that understands *both sides* and recommends crops that are viable **ecologically and economically**.

---

## 💡 Solution Overview

We propose a GenAI-powered system that:

1. **Clusters farmer data per crop** to create sustainability zones using Spectral Clustering.
2. **Matches new farm inputs** to these clusters using fuzzy similarity (z-score distance).
3. **Scores crops** using a custom profitability index from market indicators.
4. **Combines both metrics** into a composite recommendation score.
5. Uses **GPT-4** to generate human-like, data-grounded explanations.
6. Logs all decisions into a **SQLite database** for auditing and retraining.

---

## 🛠 Technologies Used

- Python (Pandas, Scikit-learn, PyTorch)
- Spectral Clustering + PCA
- Fuzzy logic (soft cluster scoring)
- OpenAI GPT-4 (explanation layer)
- SQLite (lightweight logging)
- Jupyter Notebook (interactive logic and experimentation)

---

## 📁 Project Structure

```bash
resilient-crop-ai/
├── Resilient_Crop_Agent.ipynb                # Final model with clustering + recommendation logic
├── Sustainable_Agri_AI.ipynb                 # Prior model with distributional probability matching
├── Resilient_Crop_AI_Theory_and_Innovation.docx
├── farmer_advisor_dataset.csv
├── market_researcher_dataset.csv
├── crop_advisor_v2.db                        # SQLite log (optional)
