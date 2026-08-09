# pyrefly: ignore [missing-import]
import os
import gradio as gr
import pandas as pd
from core.models import (
    load_data,
    prepare_models,
    get_data_overview,
    predict_position,
    get_classification_performance,
    predict_goals,
    get_regression_performance,
    get_model_comparison,
)

# Load data and prepare ML models
df = load_data()
models = prepare_models(df)

# Retrieve data overview components
(
    total_players, total_features, total_leagues,
    sample_df, info_df, describe_df,
    fig_bar, fig_pie, top_players_df
) = get_data_overview(df)

# Retrieve model comparison figures and summary
fig_clf, fig_mse, fig_r2, summary_md = get_model_comparison(models)

# Retrieve model performance summaries
acc_str, report_str = get_classification_performance(models)
reg_mse_str, reg_r2_str, actual_vs_pred_df = get_regression_performance(models)

# Define Event Handlers
def predict_position_wrapper(gls, ast, pk, crdy, crdr, xg, prgc, prgp, prgr):
    pred_md, fig_probs = predict_position(models, gls, ast, pk, crdy, crdr, xg, prgc, prgp, prgr)
    return pred_md, fig_probs

def predict_goals_wrapper(*val_args):
    numeric_cols = models['numeric_regression_cols']
    input_dict = dict(zip(numeric_cols, val_args))
    pred_md, r2_md, fig_coefs = predict_goals(models, input_dict)
    return pred_md, r2_md, fig_coefs


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Monochrome(), title="Players Statistics Analysis") as demo:
    gr.Markdown("# Players Statistics Analysis")
    gr.Markdown("Machine Learning analysis of top 5 European league football players")
    gr.Markdown("[View on GitHub](https://github.com/Mahmoud-islamcs/Players-Statics-Analysis-NTI_Project)")

    with gr.Tabs():
        # TAB 1: Data Overview
        with gr.Tab("Data Overview"):
            gr.Markdown("## Dataset Overview")
            with gr.Row():
                gr.Markdown(f"### Total Players\n**{total_players:,}**")
                gr.Markdown(f"### Features / Columns\n**{total_features}**")
                gr.Markdown(f"### Leagues\n**{total_leagues}**")

            gr.Markdown("### Sample Data")
            gr.Dataframe(value=sample_df, interactive=False)

            gr.Markdown("### Dataset Info")
            gr.Dataframe(value=info_df, interactive=False)

            gr.Markdown("### Descriptive Statistics")
            gr.Dataframe(value=describe_df, interactive=False)

            gr.Markdown("### Position Distribution")
            with gr.Row():
                gr.Plot(value=fig_bar)
                gr.Plot(value=fig_pie)

            gr.Markdown("### Top 10 Players by Goals")
            gr.Dataframe(value=top_players_df, interactive=False)

        # TAB 2: Position Prediction
        with gr.Tab("Position Prediction"):
            gr.Markdown("## Player Position Prediction")
            gr.Markdown("Predict a player's position (DF, FW, GK, MF) based on their statistics using **Random Forest (best model)**")

            gr.Markdown("### Enter Player Statistics")
            with gr.Row():
                with gr.Column():
                    gls = gr.Number(label="Goals (Gls)", value=5.0, minimum=0.0, maximum=50.0, step=0.5)
                    ast = gr.Number(label="Assists (Ast)", value=3.0, minimum=0.0, maximum=30.0, step=0.5)
                    pk = gr.Number(label="Penalty Kicks Made (PK)", value=0.0, minimum=0.0, maximum=20.0, step=0.5)
                with gr.Column():
                    crdy = gr.Number(label="Yellow Cards (CrdY)", value=2.0, minimum=0.0, maximum=20.0, step=0.5)
                    crdr = gr.Number(label="Red Cards (CrdR)", value=0.0, minimum=0.0, maximum=5.0, step=0.5)
                    xg = gr.Number(label="Expected Goals (xG)", value=4.0, minimum=0.0, maximum=30.0, step=0.5)
                with gr.Column():
                    prgc = gr.Number(label="Progressive Carries (PrgC)", value=50.0, minimum=0.0, maximum=300.0, step=1.0)
                    prgp = gr.Number(label="Progressive Passes (PrgP)", value=100.0, minimum=0.0, maximum=500.0, step=1.0)
                    prgr = gr.Number(label="Progressive Passes Received (PrgR)", value=60.0, minimum=0.0, maximum=400.0, step=1.0)

            predict_pos_btn = gr.Button("Predict Position", variant="primary")

            pred_pos_output = gr.Markdown()
            prob_plot_output = gr.Plot()

            predict_pos_btn.click(
                fn=predict_position_wrapper,
                inputs=[gls, ast, pk, crdy, crdr, xg, prgc, prgp, prgr],
                outputs=[pred_pos_output, prob_plot_output]
            )

            with gr.Accordion("Classification Model Performance", open=False):
                gr.Markdown(f"### {acc_str}")
                gr.Markdown("#### Classification Report:")
                gr.Code(value=report_str, language="markdown", interactive=False)

        # TAB 3: Goals Prediction
        with gr.Tab("Goals Prediction"):
            gr.Markdown("## Goals Prediction")
            gr.Markdown("Predict how many goals a player will score based on their statistics using **Linear Regression (best model)**")

            gr.Markdown("### Enter Player Statistics")
            default_vals = {
                'Rk': 100, 'Age': 25, 'Born': 1998, 'MP': 20, 'Starts': 15,
                'Min': 1500, '90s': 16, 'Ast': 3, 'PK': 0, 'PKatt': 0,
                'CrdY': 3, 'CrdR': 0, 'npxG': 3.5, 'xAG': 2.5,
                'PrgC': 50, 'PrgP': 100, 'PrgR': 60, 'Ast_90': 0.15,
                'Nation_num': 30, 'Pos_num': 1, 'Squad_num': 30,
                'Player name': 1000, 'Comp_num': 1
            }

            numeric_cols = models['numeric_regression_cols']
            reg_input_components = []

            with gr.Row():
                col1_comp = []
                col2_comp = []
                col3_comp = []

                with gr.Column():
                    for i, col_name in enumerate(numeric_cols):
                        if i % 3 == 0:
                            val = default_vals.get(col_name, 0.0)
                            comp = gr.Number(label=col_name, value=val)
                            col1_comp.append(comp)

                with gr.Column():
                    for i, col_name in enumerate(numeric_cols):
                        if i % 3 == 1:
                            val = default_vals.get(col_name, 0.0)
                            comp = gr.Number(label=col_name, value=val)
                            col2_comp.append(comp)

                with gr.Column():
                    for i, col_name in enumerate(numeric_cols):
                        if i % 3 == 2:
                            val = default_vals.get(col_name, 0.0)
                            comp = gr.Number(label=col_name, value=val)
                            col3_comp.append(comp)

            # Reconstruct list of components in order of numeric_cols
            for i, col_name in enumerate(numeric_cols):
                if i % 3 == 0:
                    reg_input_components.append(col1_comp[i // 3])
                elif i % 3 == 1:
                    reg_input_components.append(col2_comp[i // 3])
                else:
                    reg_input_components.append(col3_comp[i // 3])

            predict_goals_btn = gr.Button("Predict Goals", variant="primary")

            pred_goals_output = gr.Markdown()
            r2_metric_output = gr.Markdown()

            with gr.Accordion("Top Features for Goal Prediction", open=True):
                coef_plot_output = gr.Plot()

            with gr.Accordion("Regression Model Performance", open=False):
                with gr.Row():
                    gr.Markdown(f"### {reg_mse_str}")
                    gr.Markdown(f"### {reg_r2_str}")
                gr.Markdown("#### Sample Predictions (Actual vs Predicted)")
                gr.Dataframe(value=actual_vs_pred_df, interactive=False)

            def goals_click_handler(*vals):
                pred_md, r2_md, fig_coefs = predict_goals_wrapper(*vals)
                return pred_md, r2_md, fig_coefs

            predict_goals_btn.click(
                fn=goals_click_handler,
                inputs=reg_input_components,
                outputs=[pred_goals_output, r2_metric_output, coef_plot_output]
            )

        # TAB 4: Model Performance Comparison
        with gr.Tab("Model Performance Comparison"):
            gr.Markdown("## Model Performance Comparison")

            gr.Markdown("### Classification Models (Position Prediction)")
            gr.Plot(value=fig_clf)

            gr.Markdown("### Regression Models (Goals Prediction)")
            with gr.Row():
                gr.Plot(value=fig_mse)
                gr.Plot(value=fig_r2)

            gr.Markdown(summary_md)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
