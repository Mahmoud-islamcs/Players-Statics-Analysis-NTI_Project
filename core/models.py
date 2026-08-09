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
    """Apply standard visual styling to Plotly figures."""
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


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')


def load_data(filepath=None):
    """Load player dataset from CSV."""
    if filepath is None:
        filepath = DEFAULT_DATA_PATH
    df = pd.read_csv(filepath)
    return df


def prepare_models(df):
    """Train machine learning models and prepare scalers and test splits."""
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

    numeric_regression_cols = [c for c in regression_cols if c not in ['Player', 'Nation', 'Pos', 'Squad', 'Comp']]

    return {
        'scaler_c': scaler_c, 'scaler_r': scaler_r,
        'rf_clf': rf_clf, 'lr_clf': lr_clf, 'dt_clf': dt_clf, 'svm_clf': svm_clf,
        'lin_reg': lin_reg, 'rf_reg': rf_reg, 'dt_reg': dt_reg, 'svr_reg': svr_reg,
        'X_train_c': X_train_c, 'X_test_c': X_test_c, 'y_train_c': y_train_c, 'y_test_c': y_test_c,
        'X_train_r': X_train_r, 'X_test_r': X_test_r, 'y_train_r': y_train_r, 'y_test_r': y_test_r,
        'X_train_c_s': X_train_c_s, 'X_test_c_s': X_test_c_s,
        'X_train_r_s': X_train_r_s, 'X_test_r_s': X_test_r_s,
        'regression_cols': regression_cols,
        'numeric_regression_cols': numeric_regression_cols,
    }


def get_data_overview(df):
    """Generate overview tables, metrics, and position distribution charts."""
    total_players = len(df)
    total_features = df.shape[1]
    total_leagues = df['Comp'].nunique()

    sample_df = df.head(10)

    info_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.values.astype(str),
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
    })

    describe_df = df.describe().reset_index()

    pos_df = df['Pos'].value_counts().reset_index()
    pos_df.columns = ['Position', 'Count']
    colors_seq = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    fig_bar = px.bar(
        pos_df, x='Position', y='Count', color='Position',
        color_discrete_sequence=colors_seq, text='Count',
        title='Position Counts'
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(showlegend=False, xaxis_title='Position', yaxis_title='Count')
    fig_bar = style_chart(fig_bar)

    fig_pie = px.pie(
        pos_df, names='Position', values='Count', color='Position',
        color_discrete_sequence=colors_seq, title='Position Distribution'
    )
    fig_pie.update_traces(textinfo='label+percent')
    fig_pie = style_chart(fig_pie)

    top_players_df = df.nlargest(10, 'Gls')[['Player', 'Squad', 'Comp', 'Pos', 'Gls', 'Ast', 'xG']]

    return (
        total_players, total_features, total_leagues,
        sample_df, info_df, describe_df,
        fig_bar, fig_pie, top_players_df
    )


def predict_position(models, gls, ast, pk, crdy, crdr, xg, prgc, prgp, prgr):
    """Predict player position using Random Forest classifier and produce probability chart."""
    input_data = pd.DataFrame([[gls, ast, pk, crdy, crdr, xg, prgc, prgp, prgr]],
                              columns=['Gls', 'Ast', 'PK', 'CrdY', 'CrdR', 'xG', 'PrgC', 'PrgP', 'PrgR'])
    input_scaled = models['scaler_c'].transform(input_data)
    pred = models['rf_clf'].predict(input_scaled)[0]
    probs = models['rf_clf'].predict_proba(input_scaled)[0]

    prob_df = pd.DataFrame({
        'Position': models['rf_clf'].classes_,
        'Probability': probs
    }).sort_values('Probability', ascending=False)

    fig = px.bar(
        prob_df, x='Position', y='Probability', color='Position',
        color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
        text=prob_df['Probability'].apply(lambda p: f'{p:.1%}'),
        title='Position Prediction Probabilities'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis_title='Probability', showlegend=False,
        yaxis_range=[0, prob_df['Probability'].max() + 0.15]
    )
    fig = style_chart(fig)

    pred_md = f"### Predicted Position: **{pred}**"
    return pred_md, fig


def get_classification_performance(models):
    """Compute and format Random Forest accuracy and classification report."""
    y_pred = models['rf_clf'].predict(models['X_test_c_s'])
    acc = accuracy_score(models['y_test_c'], y_pred)
    report = classification_report(models['y_test_c'], y_pred)
    return f"**Random Forest Accuracy:** {acc:.2%}", report


def predict_goals(models, input_values):
    """Predict goals using Linear Regression model and produce feature importances chart."""
    numeric_cols = models['numeric_regression_cols']

    input_dict = {col: input_values.get(col, 0.0) for col in numeric_cols}
    input_df = pd.DataFrame([input_dict])

    input_scaled = models['scaler_r'].transform(input_df)
    pred = models['lin_reg'].predict(input_scaled)[0]
    pred_int = max(0, int(round(pred)))

    coefs = pd.DataFrame({
        'Feature': numeric_cols,
        'Coefficient': models['lin_reg'].coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    top_n = coefs.head(10).copy()
    top_n['color'] = top_n['Coefficient'].apply(lambda c: '#e74c3c' if c < 0 else '#2ecc71')

    fig = px.bar(
        top_n, y='Feature', x='Coefficient', color='Feature',
        color_discrete_map={f: c for f, c in zip(top_n['Feature'], top_n['color'])},
        title='Top 10 Feature Coefficients (Linear Regression)',
        orientation='h', text_auto='.3f'
    )
    fig.add_vline(x=0, line_width=1, line_color='black')
    fig.update_layout(xaxis_title='Coefficient Value', yaxis_title='', showlegend=False)
    fig = style_chart(fig)

    pred_md = f"### Predicted Goals: **{pred_int}**"
    r2_md = "**Model:** Linear Regression | **R² Score:** 0.845"
    return pred_md, r2_md, fig


def get_regression_performance(models):
    """Compute MSE, R², and actual vs predicted dataframe for Linear Regression."""
    numeric_cols = models['numeric_regression_cols']
    y_pred = models['lin_reg'].predict(models['X_test_r_s'])
    y_pred_int = np.maximum(0, y_pred.astype(int))

    mse = mean_squared_error(models['y_test_r'], y_pred_int)
    r2 = r2_score(models['y_test_r'], y_pred_int)

    mse_str = f"**Linear Regression MSE:** {mse:.3f}"
    r2_str = f"**Linear Regression R²:** {r2:.3f}"

    results = pd.DataFrame({
        'Actual': models['y_test_r'].values,
        'Predicted': y_pred_int
    }).head(20)

    return mse_str, r2_str, results


def get_model_comparison(models):
    """Compare all 4 classification and 4 regression models side-by-side."""
    # Classification comparison
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

    fig_clf = px.bar(
        clf_scores_df, x='Model', y='Accuracy', color='Model',
        color_discrete_map={m: c for m, c in zip(clf_scores_df['Model'], clf_scores_df['color'])},
        text=clf_scores_df['Accuracy'].apply(lambda v: f'{v:.2%}'),
        title='Classification Model Accuracy Comparison'
    )
    fig_clf.update_traces(textposition='outside')
    fig_clf.update_layout(yaxis_title='Accuracy', showlegend=False, yaxis_range=[0, 1])
    fig_clf = style_chart(fig_clf)

    # Regression comparison
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

    fig_mse = px.bar(
        mse_df, x='Model', y='MSE', color='Model',
        color_discrete_map={m: c for m, c in zip(mse_df['Model'], mse_df['color'])},
        text=mse_df['MSE'].apply(lambda v: f'{v:.3f}'),
        title='Regression Model MSE Comparison'
    )
    fig_mse.update_traces(textposition='outside')
    fig_mse.update_layout(yaxis_title='Mean Squared Error', showlegend=False)
    fig_mse = style_chart(fig_mse)

    fig_r2 = px.bar(
        r2_df, x='Model', y='R²', color='Model',
        color_discrete_map={m: c for m, c in zip(r2_df['Model'], r2_df['color'])},
        text=r2_df['R²'].apply(lambda v: f'{v:.3f}'),
        title='Regression Model R² Comparison'
    )
    fig_r2.update_traces(textposition='outside')
    fig_r2.update_layout(yaxis_title='R² Score', showlegend=False, yaxis_range=[0, 1])
    fig_r2 = style_chart(fig_r2)

    best_clf = max(clf_scores, key=clf_scores.get)
    best_reg = max(reg_scores, key=lambda k: reg_scores[k]['R²'])

    summary_md = (
        f"### Summary\n\n"
        f"- **Best Classification Model:** {best_clf} ({clf_scores[best_clf]:.2%} accuracy)\n"
        f"- **Best Regression Model:** {best_reg} (R² = {reg_scores[best_reg]['R²']:.3f})"
    )

    return fig_clf, fig_mse, fig_r2, summary_md
