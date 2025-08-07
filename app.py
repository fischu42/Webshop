import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import yaml
from pycaret.regression import load_model, predict_model
import webbrowser
import threading

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
    'pixel_w', 'pixel_h', 'back_camera_resolution',
    'front_camera_resolution', 'w', 'h', 
    'd', 'mass', 'battery'
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
    html.H2("📱 Smartphone Price Predictor", className="my-4 text-center"),

    dbc.Card([
        dbc.CardHeader("🔧 Inputs"),
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
        dbc.CardBody(
            html.Div(id='output-prediction', style={'fontSize': 24, 'fontWeight': 'bold'})
        )
    ])
], fluid=True)

# ---------------------
# Callback
# ---------------------
@app.callback(
    Output('output-prediction', 'children'),
    Output('input-warning', 'children'),
    Output('input-warning', 'is_open'),
    Input('predict-button', 'n_clicks'),
    [
        State(f'input-{col}', 'value') for col in (binary_cat + ordinal_cat + numeric_columns)
    ]
)
def make_prediction(n_clicks, *values):
    if n_clicks == 0:
        return "", "", False

    all_cols = binary_cat + ordinal_cat + numeric_columns
    input_dict = dict(zip(all_cols, values))

    # Check for missing values
    missing = [key for key, val in input_dict.items() if val is None]
    if missing:
        return "", f"Please fill in all inputs. Missing: {', '.join(missing)}", True

    # Min-max normalize numeric features
    for col in numeric_columns:
        val = input_dict[col]
        min_val = stats[col]['min']
        max_val = stats[col]['max']
        input_dict[col] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0

    input_df = pd.DataFrame([input_dict])
    prediction = predict_model(model, data=input_df)

    # Get prediction column
    prediction_columns = prediction.columns.difference(input_df.columns)
    if prediction_columns.empty:
        return "", "Prediction failed: No output column returned by model.", True

    pred_col = prediction_columns[0]
    pred = prediction[pred_col].iloc[0]

    return f"Predicted price: {pred:.2f}", "", False

# ---------------------
# Auto-open browser on launch
# ---------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)
