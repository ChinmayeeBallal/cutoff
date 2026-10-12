# 🎓 Cutoff Score Predictor

An end-to-end Machine Learning pipeline and Streamlit application to predict university admission cutoff marks and ranks.

## 🚀 Features
- **Interactive Predictor UI:** Simple select boxes and input fields to quickly estimate cutoff targets.
- **Optimized Model Pipeline:** Features a `RandomForestRegressor` pipeline pre-trained with optimized hyperparameters ($R^2 \approx 0.85$).
- **Unified Processing:** Categorical values are automatically processed using robust missing-value imputation and one-hot encoding on the fly.

## 📁 Directory Structure
- `app.py`: Streamlit dashboard code.
- `requirements.txt`: Python package dependencies.
- `best_model_pipeline.pkl`: Serialized scikit-learn random forest pipeline.
- `scaler.pkl`: Numeric scaling components.
- `cutoffs.csv`: Raw historical cutoff data used for prediction.

## 🛠️ Local Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ChinmayeeBallal/cutoff.git
   cd cutoff
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit App**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment
This project is configured for seamless deployment to the [Streamlit Community Cloud](https://share.streamlit.io/). Just link this repository to your Streamlit Cloud workspace, set the entry point to `app.py`, and you're live!
