from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import json
import numpy as np

# --- 1. Initialize Flask App and Load Resources ---
app = Flask(__name__)
class MatchPredictor(nn.Module):
    def __init__(self, num_features):
        super(MatchPredictor, self).__init__(); self.layer_1 = nn.Linear(num_features, 64); self.layer_2 = nn.Linear(64, 64); self.output_layer = nn.Linear(64, 1); self.relu = nn.ReLU(); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.layer_1(x)); x = self.relu(self.layer_2(x)); x = self.output_layer(x); x = self.sigmoid(x); return x

def load_resources():
    print("Loading resources..."); num_features = 5; model = MatchPredictor(num_features=num_features);
    model = torch.jit.load('dota2_predictor.pt')
    model.eval()
    with open('team_data.json', 'r') as f: team_data = json.load(f)
    with open('scaler.json', 'r') as f: scaler_data = json.load(f); scaler_mean = np.array(scaler_data['mean']); scaler_scale = np.array(scaler_data['scale'])
    print("Resources loaded successfully."); return model, team_data, scaler_mean, scaler_scale

MODEL, TEAM_DATA, SCALER_MEAN, SCALER_SCALE = load_resources()

# --- 2. Define Prediction Logic ---
def predict_match(team_a, team_b):
    team_a_ratings = TEAM_DATA[team_a]; team_b_ratings = TEAM_DATA[team_b]
    feature_vector = np.array([[team_a_ratings['elo32'] - team_b_ratings['elo32'], team_a_ratings['elo64'] - team_b_ratings['elo64'], team_a_ratings['glicko_mu'] - team_b_ratings['glicko_mu'], team_a_ratings['glicko_rd'], team_b_ratings['glicko_rd']]])
    feature_vector_scaled = (feature_vector - SCALER_MEAN) / SCALER_SCALE
    with torch.no_grad(): input_tensor = torch.tensor(feature_vector_scaled, dtype=torch.float32); win_prob_a = MODEL(input_tensor).item()
    return win_prob_a

# --- 3. Define Web Routes (Endpoints) ---
@app.route('/')
def index():
    team_list = sorted(TEAM_DATA.keys()); return render_template('index.html', team_list=team_list)

@app.route('/simulate', methods=['POST'])
def simulate():
    """Receives selected teams, runs FULL simulation, and returns results."""
    data = request.get_json(); selected_teams = data.get('teams', [])
    if len(selected_teams) != 8: return jsonify({'error': 'Please select 8 teams.'}), 400

    results = {}

    # --- Round 1: Upper Bracket ---
    ub_r2_teams = []; lb_r1_teams = []
    for i in range(4):
        t1, t2 = selected_teams[i*2], selected_teams[i*2+1]; prob = predict_match(t1, t2)
        winner, loser = (t1, t2) if prob > 0.5 else (t2, t1)
        results[f'ub_r1_m{i}'] = {'teams': [t1, t2], 'prob': prob, 'winner': winner, 'loser': loser}
        ub_r2_teams.append(winner); lb_r1_teams.append(loser)

    # --- Round 2: Lower Bracket ---
    lb_r2_from_lb_winners = []
    for i in range(2):
        t1, t2 = lb_r1_teams[i*2], lb_r1_teams[i*2+1]; prob = predict_match(t1, t2)
        winner, loser = (t1, t2) if prob > 0.5 else (t2, t1)
        results[f'lb_r1_m{i}'] = {'teams': [t1, t2], 'prob': prob, 'winner': winner, 'loser': loser}
        lb_r2_from_lb_winners.append(winner)

    # --- Round 3: Upper Bracket Semifinals ---
    ub_final_teams = []; lb_r2_from_ub_losers = []
    for i in range(2):
        t1, t2 = ub_r2_teams[i*2], ub_r2_teams[i*2+1]; prob = predict_match(t1, t2)
        winner, loser = (t1, t2) if prob > 0.5 else (t2, t1)
        results[f'ub_r2_m{i}'] = {'teams': [t1, t2], 'prob': prob, 'winner': winner, 'loser': loser}
        ub_final_teams.append(winner); lb_r2_from_ub_losers.append(loser)

    # --- Round 4: Lower Bracket Quarterfinals ---
    lb_semifinal_teams = []
    for i in range(2):
        t1, t2 = lb_r2_from_ub_losers[i], lb_r2_from_lb_winners[i]; prob = predict_match(t1, t2)
        winner, loser = (t1, t2) if prob > 0.5 else (t2, t1)
        results[f'lb_r2_m{i}'] = {'teams': [t1, t2], 'prob': prob, 'winner': winner, 'loser': loser}
        lb_semifinal_teams.append(winner)

    # --- Round 5: Upper Bracket Final ---
    t1, t2 = ub_final_teams[0], ub_final_teams[1]; prob = predict_match(t1, t2)
    ub_winner, ub_loser = (t1, t2) if prob > 0.5 else (t2, t1)
    results['ub_final_m0'] = {'teams': [t1, t2], 'prob': prob, 'winner': ub_winner, 'loser': ub_loser}

    # --- Round 6: Lower Bracket Semifinal ---
    t1, t2 = lb_semifinal_teams[0], lb_semifinal_teams[1]; prob = predict_match(t1, t2)
    lb_final_qualifier, _ = (t1, t2) if prob > 0.5 else (t2, t1)
    results['lb_semi_m0'] = {'teams': [t1, t2], 'prob': prob, 'winner': lb_final_qualifier}

    # --- Round 7: Lower Bracket Final ---
    t1, t2 = ub_loser, lb_final_qualifier; prob = predict_match(t1, t2)
    lb_winner, _ = (t1, t2) if prob > 0.5 else (t2, t1)
    results['lb_final_m0'] = {'teams': [t1, t2], 'prob': prob, 'winner': lb_winner}

    # --- Round 8: Grand Final ---
    t1, t2 = ub_winner, lb_winner; prob = predict_match(t1, t2)
    champion, _ = (t1, t2) if prob > 0.5 else (t2, t1)
    results['grand_final_m0'] = {'teams': [t1, t2], 'prob': prob, 'winner': champion}
    results['champion'] = champion

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
