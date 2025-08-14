import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import yaml
from pycaret.regression import load_model, predict_model
import webbrowser
import threading
import shap
import plotly.graph_objects as go
import numpy as np
import os

# --------------------
# Load model and stats
# --------------------
model = load_model('Models/best_regression_model')

with open('numeric_stats.yaml', 'r') as f:
    stats = yaml.safe_load(f)

# ---------------------
# Define feature groups
# ---------------------
binary_cat = ['type', 'face_id', 'dual_sim']
ordinal_cat = ['back_camera']
numeric_columns = [
 'capacity', 'memory', 'screen_diagonal', 
 'aspect_ratio', 'pixel_count', 'back_camera_resolution',
 'front_camera_resolution', 'volume', 'mass', 'battery'
 ]

binary_options = [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]
ordinal_options = [{'label': str(i), 'value': i} for i in range(1, 5)]

# ---------------------
# Dash App
# ---------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PyCaret Regression App"

def create_input_row(label, input_component):
    return dbc.Row([
        dbc.Col(html.Label(label), width=4),
        dbc.Col(input_component, width=8)
    ], className="mb-2")

app.layout = dbc.Container([
    html.H2("Smartphone Price Predictor", className="my-4 text-center"),

    dbc.Card([
        dbc.CardHeader("Inputs"),
        dbc.CardBody([
            html.H5("Binary Categorical Inputs"),
            *[
                create_input_row(col, dcc.Dropdown(options=binary_options, id=f'input-{col}', placeholder="Select..."))
                for col in binary_cat
            ],
            html.Hr(),

            html.H5("Ordinal Input"),
            create_input_row("back_camera", dcc.Dropdown(options=ordinal_options, id='input-back_camera', placeholder="Select...")),
            html.Hr(),

            html.H5("Numeric Inputs"),
            *[
                create_input_row(col, dcc.Input(type='number', id=f'input-{col}', placeholder="Enter value..."))
                for col in numeric_columns
            ],

            dbc.Alert(id="input-warning", color="danger", is_open=False, dismissable=True, className="mt-3"),

            dbc.Button("Predict", id='predict-button', color="primary", className="mt-4"),
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Prediction Output"),
        dbc.CardBody([
            html.Div(id='output-prediction', style={'fontSize': 24, 'fontWeight': 'bold'}),
            html.Hr(),
            html.Div([dcc.Graph(id='shap-waterfall')])
        ])
    ])
], fluid=True)

# ---------------------
# Callback
# ---------------------
@app.callback(
    Output('output-prediction', 'children'),
    Output('input-warning', 'children'),
    Output('input-warning', 'is_open'),
    Output('shap-waterfall', 'figure'),
    Input('predict-button', 'n_clicks'),
    [
        State(f'input-{col}', 'value') for col in (binary_cat + ordinal_cat + numeric_columns)
    ]
)
def make_prediction(n_clicks, *values):
    if n_clicks == 0:
        return "", "", False, go.Figure()

    all_cols = binary_cat + ordinal_cat + numeric_columns
    input_dict = dict(zip(all_cols, values))

    # Check for missing values
    missing = [key for key, val in input_dict.items() if val is None]
    if missing:
        return "", f"Please fill in all inputs. Missing: {', '.join(missing)}", True, go.Figure()

    # Min-max normalize numeric features
    for col in numeric_columns:
        val = input_dict[col]
        min_val = stats[col]['min']
        max_val = stats[col]['max']
        input_dict[col] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0

    input_df = pd.DataFrame([input_dict])
    prediction = predict_model(model, data=input_df)

    prediction_columns = prediction.columns.difference(input_df.columns)
    if prediction_columns.empty:
        return "", "Prediction failed: No output column returned by model.", True, go.Figure()

    pred_col = prediction_columns[0]
    pred = prediction[pred_col].iloc[0]

    # ---------------------
    # SHAP explanation
    # ---------------------
    # Preprocess data using the pipeline's steps before the actual estimator
    preprocessing_steps = model[:-1]  # everything except last step
    X_transformed = preprocessing_steps.transform(input_df)

    # Extract raw trained estimator
    raw_model = model.named_steps['actual_estimator']

    # TreeExplainer for tree-based models like ExtraTrees
    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(X_transformed)

    # Waterfall plot using transformed feature names
    feature_names = preprocessing_steps.transform(pd.DataFrame([input_dict])).columns \
        if hasattr(X_transformed, 'columns') else [f'feature_{i}' for i in range(X_transformed.shape[1])]

    shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_pairs = sorted(zip(feature_names, shap_array[0]), key=lambda x: abs(x[1]), reverse=True)

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(shap_pairs),
        x=[f[0] for f in shap_pairs],
        text=[f"{f[1]:.2f}" for f in shap_pairs],
        y=[f[1] for f in shap_pairs]
    ))
    fig.update_layout(title="SHAP Waterfall Plot", waterfallgap=0.3)

    return f"Predicted price: {pred:.2f}", "", False, fig


# ---------------------
# Auto-open browser on launch
# ---------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")


if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.25, open_browser).start()
    app.run(debug=True)
