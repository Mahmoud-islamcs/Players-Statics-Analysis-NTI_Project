import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def style_chart(fig, height=550):
    fig.update_layout(
        height=height,
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14,
        legend_font_size=14,
        font_family="Arial, sans-serif",
    )
    return fig

st.set_page_config(page_title="Players Statistics Analysis", layout="wide")

st.title("Players Statistics Analysis")
st.markdown("Machine Learning analysis of top 5 European league football players")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_data.csv')
    return df

df = load_data()

with st.sidebar:
    st.title("Navigation")
    page = option_menu(
        menu_title=None,
        options=["Data Overview", "Position Prediction", "Goals Prediction", "Model Performance Comparison"],
        icons=["table", "bullseye", "graph-up-arrow", "bar-chart-steps"],
        default_index=0,
        styles={
            "nav-link": {"font-size": "13px", "padding": "0.3rem 0.5rem", "margin": "0rem"},
            "icon": {"font-size": "13px"},
        },
    )
    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/Mahmoud-islamcs/Players-Statics-Analysis-NTI_Project" '
        'target="_blank" style="text-decoration:none;">'
        '<b>View on GitHub</b></a>',
        unsafe_allow_html=True,
    )

@st.cache_resource
def prepare_models():
    X_class = df[['Gls', 'Ast', 'PK', 'CrdY', 'CrdR', 'xG', 'PrgC', 'PrgP', 'PrgR']]
    y_class = df['Pos']

    regression_cols = [c for c in df.columns if c not in [
        'Player', 'Nation', 'Comp', 'Pos', 'Squad', 'Gls', 'G+A', 'G-PK',
        'xG', 'npxG+xAG', 'G+A_90', 'G-PK_90', 'G+A-PK_90', 'xG_90',
        'Gls_90', 'xG+xAG_90', 'npxG_90', 'npxG+xAG_90'
    ]]
    X_reg = df[regression_cols]
    y_reg = df['Gls']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    scaler_c = StandardScaler()
    X_train_c_s = scaler_c.fit_transform(X_train_c)
    X_test_c_s = scaler_c.transform(X_test_c)

    scaler_r = StandardScaler()
    X_train_r_s = scaler_r.fit_transform(X_train_r)
    X_test_r_s = scaler_r.transform(X_test_r)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_c_s, y_train_c)
    lr_clf = LogisticRegression(random_state=42).fit(X_train_c_s, y_train_c)
    dt_clf = DecisionTreeClassifier(random_state=42).fit(X_train_c_s, y_train_c)
    svm_clf = SVC(kernel='rbf', random_state=42).fit(X_train_c_s, y_train_c)

    lin_reg = LinearRegression().fit(X_train_r_s, y_train_r)
    rf_reg = RandomForestRegressor(random_state=42).fit(X_train_r_s, y_train_r)
    dt_reg = DecisionTreeRegressor(random_state=42).fit(X_train_r_s, y_train_r)
    svr_reg = SVR(kernel='rbf').fit(X_train_r_s, y_train_r)

    return {
        'scaler_c': scaler_c, 'scaler_r': scaler_r,
        'rf_clf': rf_clf, 'lr_clf': lr_clf, 'dt_clf': dt_clf, 'svm_clf': svm_clf,
        'lin_reg': lin_reg, 'rf_reg': rf_reg, 'dt_reg': dt_reg, 'svr_reg': svr_reg,
        'X_train_c': X_train_c, 'X_test_c': X_test_c, 'y_train_c': y_train_c, 'y_test_c': y_test_c,
        'X_train_r': X_train_r, 'X_test_r': X_test_r, 'y_train_r': y_train_r, 'y_test_r': y_test_r,
        'X_train_c_s': X_train_c_s, 'X_test_c_s': X_test_c_s,
        'X_train_r_s': X_train_r_s, 'X_test_r_s': X_test_r_s,
        'regression_cols': regression_cols,
    }

models = prepare_models()

if page == "Data Overview":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Players", f"{len(df):,}")
    col2.metric("Features/Columns", df.shape[1])
    col3.metric("Leagues", df['Comp'].nunique())

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Info")
    buf = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
    })
    st.dataframe(buf, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Position Distribution")
    pos_df = df['Pos'].value_counts().reset_index()
    pos_df.columns = ['Position', 'Count']
    colors_seq = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(pos_df, x='Position', y='Count', color='Position',
                         color_discrete_sequence=colors_seq, text='Count',
                         title='Position Counts')
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, xaxis_title='Position', yaxis_title='Count')
        st.plotly_chart(style_chart(fig_bar), use_container_width=True)
    with col_b:
        fig_pie = px.pie(pos_df, names='Position', values='Count', color='Position',
                         color_discrete_sequence=colors_seq, title='Position Distribution')
        fig_pie.update_traces(textinfo='label+percent')
        st.plotly_chart(style_chart(fig_pie), use_container_width=True)

    st.subheader("Top 10 Players by Goals")
    top_players = df.nlargest(10, 'Gls')[['Player', 'Squad', 'Comp', 'Pos', 'Gls', 'Ast', 'xG']]
    st.dataframe(top_players, use_container_width=True)

elif page == "Position Prediction":
    st.header("Player Position Prediction")
    st.markdown("Predict a player's position (DF, FW, GK, MF) based on their statistics using **Random Forest (best model)**")

    st.subheader("Enter Player Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        gls = st.number_input("Goals (Gls)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        ast = st.number_input("Assists (Ast)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
        pk = st.number_input("Penalty Kicks Made (PK)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    with col2:
        crdy = st.number_input("Yellow Cards (CrdY)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        crdr = st.number_input("Red Cards (CrdR)", min_value=0.0, max_value=5.0, value=0.0, step=0.5)
        xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
    with col3:
        prgc = st.number_input("Progressive Carries (PrgC)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
        prgp = st.number_input("Progressive Passes (PrgP)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
        prgr = st.number_input("Progressive Passes Received (PrgR)", min_value=0.0, max_value=400.0, value=60.0, step=1.0)

    if st.button("Predict Position", type="primary"):
        input_data = pd.DataFrame([[gls, ast, pk, crdy, crdr, xg, prgc, prgp, prgr]],columns=['Gls', 'Ast', 'PK', 'CrdY', 'CrdR', 'xG', 'PrgC', 'PrgP', 'PrgR'])
        input_scaled = models['scaler_c'].transform(input_data)
        pred = models['rf_clf'].predict(input_scaled)[0]
        probs = models['rf_clf'].predict_proba(input_scaled)[0]

        st.success(f"### Predicted Position: {pred}")

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Position': models['rf_clf'].classes_,
            'Probability': probs
        }).sort_values('Probability', ascending=False)
        fig = px.bar(prob_df, x='Position', y='Probability', color='Position',
                     color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                     text=prob_df['Probability'].apply(lambda p: f'{p:.1%}'),
                     title='Position Prediction Probabilities')
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis_title='Probability', showlegend=False,
                          yaxis_range=[0, prob_df['Probability'].max() + 0.15])
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with st.expander("Classification Model Performance"):
        y_pred = models['rf_clf'].predict(models['X_test_c_s'])
        acc = accuracy_score(models['y_test_c'], y_pred)
        st.metric("Random Forest Accuracy", f"{acc:.2%}")
        st.text("Classification Report:")
        st.code(classification_report(models['y_test_c'], y_pred))

elif page == "Goals Prediction":
    st.header("Goals Prediction")
    st.markdown("Predict how many goals a player will score based on their statistics using **Linear Regression (best model)**")

    st.subheader("Enter Player Statistics")
    reg_cols = models['regression_cols']
    inputs = {}
    col1, col2, col3 = st.columns(3)
    default_vals = {
        'Rk': 100, 'Age': 25, 'Born': 1998, 'MP': 20, 'Starts': 15,
        'Min': 1500, '90s': 16, 'Ast': 3, 'PK': 0, 'PKatt': 0,
        'CrdY': 3, 'CrdR': 0, 'npxG': 3.5, 'xAG': 2.5,
        'PrgC': 50, 'PrgP': 100, 'PrgR': 60, 'Ast_90': 0.15,
        'Nation_num': 30, 'Pos_num': 1, 'Squad_num': 30,
        'Player name': 1000, 'Comp_num': 1
    }
    numeric_cols = [c for c in reg_cols if c not in ['Player', 'Nation', 'Pos', 'Squad', 'Comp']]
    for i, col in enumerate(numeric_cols):
        target_col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
        default = default_vals.get(col, 0.0)
        if col in ['Rk', 'MP', 'Starts', 'Min', '90s', 'PrgC', 'PrgP', 'PrgR']:
            inputs[col] = target_col.number_input(col, value=int(default), step=1)
        else:
            inputs[col] = target_col.number_input(col, value=float(default), step=0.1, format="%.2f")

    if st.button("Predict Goals", type="primary"):
        input_df = pd.DataFrame([inputs])
        input_scaled = models['scaler_r'].transform(input_df)
        pred = models['lin_reg'].predict(input_scaled)[0]
        pred = max(0, int(round(pred)))
        st.success(f"### Predicted Goals: **{pred}**")
        st.metric("Model", "Linear Regression", delta=f"R² = 0.845")

        with st.expander("Top Features for Goal Prediction"):
            coefs = pd.DataFrame({
                'Feature': numeric_cols,
                'Coefficient': models['lin_reg'].coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            top_n = coefs.head(10).copy()
            top_n['color'] = top_n['Coefficient'].apply(lambda c: '#e74c3c' if c < 0 else '#2ecc71')
            fig = px.bar(top_n, y='Feature', x='Coefficient', color='Feature',
                         color_discrete_map={f: c for f, c in zip(top_n['Feature'], top_n['color'])},
                         title='Top 10 Feature Coefficients (Linear Regression)',
                         orientation='h', text_auto='.3f')
            fig.add_vline(x=0, line_width=1, line_color='black')
            fig.update_layout(xaxis_title='Coefficient Value', yaxis_title='', showlegend=False)
            st.plotly_chart(style_chart(fig), use_container_width=True)

    with st.expander("Regression Model Performance"):
        y_pred = models['lin_reg'].predict(models['X_test_r_s'])
        y_pred_int = np.maximum(0, y_pred.astype(int))
        mse = mean_squared_error(models['y_test_r'], y_pred_int)
        r2 = r2_score(models['y_test_r'], y_pred_int)
        col1, col2 = st.columns(2)
        col1.metric("Linear Regression MSE", f"{mse:.3f}")
        col2.metric("Linear Regression R²", f"{r2:.3f}")

        results = pd.DataFrame({
            'Actual': models['y_test_r'].values,
            'Predicted': y_pred_int
        }).head(20)
        st.dataframe(results, use_container_width=True)

elif page == "Model Performance Comparison":
    st.header("Model Performance Comparison")

    st.subheader("Classification Models (Position Prediction)")
    clf_models = {
        'Random Forest': models['rf_clf'],
        'Logistic Regression': models['lr_clf'],
        'Decision Tree': models['dt_clf'],
        'SVM (RBF)': models['svm_clf'],
    }
    clf_scores = {}
    for name, model in clf_models.items():
        y_pred = model.predict(models['X_test_c_s'])
        clf_scores[name] = accuracy_score(models['y_test_c'], y_pred)

    clf_scores_df = pd.DataFrame(list(clf_scores.items()), columns=['Model', 'Accuracy'])
    clf_scores_df['color'] = clf_scores_df['Accuracy'].apply(
        lambda v: '#2ecc71' if v == clf_scores_df['Accuracy'].max() else '#3498db')
    fig = px.bar(clf_scores_df, x='Model', y='Accuracy', color='Model',
                 color_discrete_map={m: c for m, c in zip(clf_scores_df['Model'], clf_scores_df['color'])},
                 text=clf_scores_df['Accuracy'].apply(lambda v: f'{v:.2%}'),
                 title='Classification Model Accuracy Comparison')
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis_title='Accuracy', showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Regression Models (Goals Prediction)")
    reg_models = {
        'Linear Regression': models['lin_reg'],
        'Random Forest': models['rf_reg'],
        'Decision Tree': models['dt_reg'],
        'SVR (RBF)': models['svr_reg'],
    }
    reg_scores = {}
    for name, model in reg_models.items():
        y_pred = model.predict(models['X_test_r_s'])
        y_pred_int = np.maximum(0, y_pred.astype(int))
        reg_scores[name] = {
            'MSE': mean_squared_error(models['y_test_r'], y_pred_int),
            'R²': r2_score(models['y_test_r'], y_pred_int),
        }

    mse_df = pd.DataFrame([(k, v['MSE']) for k, v in reg_scores.items()], columns=['Model', 'MSE'])
    r2_df = pd.DataFrame([(k, v['R²']) for k, v in reg_scores.items()], columns=['Model', 'R²'])

    mse_df['color'] = mse_df['MSE'].apply(lambda v: '#2ecc71' if v == mse_df['MSE'].min() else '#e74c3c')
    r2_df['color'] = r2_df['R²'].apply(lambda v: '#2ecc71' if v == r2_df['R²'].max() else '#3498db')

    col1, col2 = st.columns(2)
    with col1:
        fig_mse = px.bar(mse_df, x='Model', y='MSE', color='Model',
                         color_discrete_map={m: c for m, c in zip(mse_df['Model'], mse_df['color'])},
                         text=mse_df['MSE'].apply(lambda v: f'{v:.3f}'),
                         title='Regression Model MSE Comparison')
        fig_mse.update_traces(textposition='outside')
        fig_mse.update_layout(yaxis_title='Mean Squared Error', showlegend=False)
        st.plotly_chart(style_chart(fig_mse), use_container_width=True)
    with col2:
        fig_r2 = px.bar(r2_df, x='Model', y='R²', color='Model',
                        color_discrete_map={m: c for m, c in zip(r2_df['Model'], r2_df['color'])},
                        text=r2_df['R²'].apply(lambda v: f'{v:.3f}'),
                        title='Regression Model R² Comparison')
        fig_r2.update_traces(textposition='outside')
        fig_r2.update_layout(yaxis_title='R² Score', showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(style_chart(fig_r2), use_container_width=True)

    st.subheader("Summary")
    best_clf = max(clf_scores, key=clf_scores.get)
    best_reg = max(reg_scores, key=lambda k: reg_scores[k]['R²'])
    col1, col2 = st.columns(2)
    col1.success(f"**Best Classification Model:** {best_clf} ({clf_scores[best_clf]:.2%} accuracy)")
    col2.success(f"**Best Regression Model:** {best_reg} (R² = {reg_scores[best_reg]['R²']:.3f})")
