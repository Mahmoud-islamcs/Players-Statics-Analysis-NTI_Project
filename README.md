---
title: Players Statistics Analysis
emoji: ⚽
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Players Statistics Analysis

Analyze top-5 European league footballers and predict player position & goal output using machine learning, all served through an interactive Gradio dashboard.

---

## Team

**Aya Kamal &middot; Lama Nezar &middot; Khaled Zakria &middot; Mahmoud Islam &middot; Adham Mohammed**

> *"Let data guide your scouting decisions."*

---

## Features

- **Data Overview** — Explore the cleaned dataset, position distribution, and top performers
- **Position Prediction** — Classify a player's role (DF / MF / FW / GK) from their stats (Random Forest, 75.7% accuracy)
- **Goals Prediction** — Forecast total goals scored using linear regression (R&#178; = 0.845)
- **Model Comparison** — Compare 4 classification and 4 regression models side-by-side

## Data Workflow

1. **Data Cleaning** — Removed inaccuracies, handled missing values, ensured consistency across metrics
2. **Exploration** — Analyzed league-wide trends (goal distribution, player positions, age breakdown, disciplinary records)
3. **Visualization** — Built interactive Power BI dashboards for top players, goal/assist breakdowns, and scouting comparisons
4. **Machine Learning** — Trained and evaluated classification and regression models to predict position and goal output

## Results

| Task | Best Model | Metric | Score |
|------|-----------|--------|-------|
| Position Prediction | Random Forest | Accuracy | 75.7% |
| Goals Prediction | Linear Regression | R&#178; | 0.845 |

**Top 5 identified players:** Kylian Mbappe &middot; Harry Kane &middot; Cole Palmer &middot; Artem Dovbyk &middot; Lautaro Martinez

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?logo=gradio)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3cb371?logo=plotly)
![scikit-learn](https://img.shields.io/badge/scikit_learn-1.6-orange?logo=scikit-learn)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?logo=powerbi&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)

## Project Structure

```
├── core/
│   ├── __init__.py         # Core package initialization
│   └── models.py           # ML model training, evaluation & inference logic
├── app.py                  # Gradio web application
├── requirements.txt        # Python dependencies
├── data/
│   ├── cleaned_data.csv    # Preprocessed player statistics
│   └── top5-players.csv    # Raw dataset (top 5 leagues)
├── notebooks/
│   └── Final Code.ipynb    # Full EDA & model training notebook
├── docs/
│   └── Machine Learnging Analysis of player stastics_compressed.pdf.pdf
└── README.md
```

## Quick Start

```bash
git clone https://github.com/Mahmoud-islamcs/Players-Statics-Analysis-NTI_Project.git
cd Players-Statics-Analysis-NTI_Project
pip install -r requirements.txt
python app.py
```

## Feedback

If you are a football fan, scout, or data enthusiast, we would love to hear your thoughts or collaborate.
