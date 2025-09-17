import requests
import pandas as pd
import time
import math
from glicko2 import Glicko2
import json
from datetime import datetime

BASE_URL = "https://api.opendota.com/api"
DEFAULT_ELO = 1500

def format_timestamp(unix_timestamp):
    datetime_object = datetime.fromtimestamp(unix_timestamp)
    return datetime_object.strftime("%Y-%m-%d %H:%M:%S")

def get_teams_map(path='teams.json'):
    with open(path, "r") as f:
        teams_map_with_string_keys = json.load(f)
        teams_map = {int(k): v for k, v in teams_map_with_string_keys.items()}
        return teams_map


def get_matches_from_leagues(num_leagues=100):
    """Fetches matches from a specified number of top-tier Dota 2 leagues."""
    print("\nFetching list of all leagues from OpenDota...")
    leagues_response = requests.get(f"{BASE_URL}/leagues")
    leagues_response.raise_for_status()
    all_leagues = leagues_response.json()

    professional_leagues = [
        # lg for lg in all_leagues if lg.get('tier') in ['premium', 'professional']
        lg for lg in all_leagues if lg.get('tier') in ['premium']
    ]
    professional_leagues.sort(key=lambda x: x['leagueid'], reverse=True)

    leagues_to_fetch = professional_leagues[:num_leagues]
    print(f"Selected the top {len(leagues_to_fetch)} professional leagues to process.")

    all_matches = []

    for league in leagues_to_fetch:
        league_id = league['leagueid']
        league_name = league['name']
        print(f"Fetching matches for league: '{league_name}' (ID: {league_id})")

        try:
            matches_response = requests.get(f"{BASE_URL}/leagues/{league_id}/matches")
            matches_response.raise_for_status()
            matches_for_league = matches_response.json()
            for match in matches_for_league:
                match['league_name'] = league_name
            all_matches.extend(matches_for_league)
            print(f"  > Found {len(matches_for_league)} matches.")
        except requests.exceptions.HTTPError as e:
            print(f"  > Could not fetch matches for league {league_id}: {e}")
        time.sleep(1.1)

    print(f"\nTotal matches fetched from all leagues: {len(all_matches)}")
    if not all_matches:
        return []

    all_matches.sort(key=lambda x: x['start_time'])
    print("All matches have been sorted chronologically.")
    return all_matches

def create_match_dataset_multi_rating(matches, teams_map):
    """
    Processes matches to create a dataset with multiple historical rating systems.
    """
    # --- NEW: Initialize Glicko-2 and multiple Elo ratings ---
    env = Glicko2(tau=0.5)
    team_ratings = {} # Will store dicts like {team_id: {'elo32': 1500, 'elo64': 1500, 'glicko2': Glicko2Player}}
    training_data = []

    print("\nProcessing matches to calculate multiple historical ratings...")
    for i, match in enumerate(matches):
        if 'radiant_team_id' not in match or 'dire_team_id' not in match:
            continue

        rad_id = match['radiant_team_id']
        dir_id = match['dire_team_id']

        # --- NEW: Get or initialize full rating profiles for both teams ---
        if rad_id not in team_ratings:
            team_ratings[rad_id] = {'elo32': 1500, 'elo64': 1500, 'glicko2': env.create_rating()}
        if dir_id not in team_ratings:
            team_ratings[dir_id] = {'elo32': 1500, 'elo64': 1500, 'glicko2': env.create_rating()}

        rad_ratings_before = team_ratings[rad_id]
        dir_ratings_before = team_ratings[dir_id]

        radiant_win = 1 if match['radiant_win'] else 0

        # Look up team names
        radiant_name = teams_map.get(rad_id, "Unknown Team")
        dire_name = teams_map.get(dir_id, "Unknown Team")

        # --- NEW: Append all rating systems to the dataset ---
        training_data.append({
            'match_id': match['match_id'],
            'date_time': format_timestamp(match['start_time']),
            'start_time': match['start_time'],
            'league_id': match['leagueid'],
            'league_name': match['league_name'],
            'radiant_name': radiant_name,
            'dire_name': dire_name,
            'radiant_elo32_before': rad_ratings_before['elo32'],
            'dire_elo32_before': dir_ratings_before['elo32'],
            'radiant_elo64_before': rad_ratings_before['elo64'],
            'dire_elo64_before': dir_ratings_before['elo64'],
            'radiant_glicko_mu_before': rad_ratings_before['glicko2'].mu,
            'dire_glicko_mu_before': dir_ratings_before['glicko2'].mu,
            'radiant_glicko_rd_before': rad_ratings_before['glicko2'].phi, # RD is phi in this library
            'dire_glicko_rd_before': dir_ratings_before['glicko2'].phi,
            'radiant_win': radiant_win
        })

        # --- NEW: Calculate updated ratings for all systems ---
        # Elo with K=32
        new_rad_elo32, new_dir_elo32 = calculate_elo(rad_ratings_before['elo32'], dir_ratings_before['elo32'], radiant_win, k=32)
        # Elo with K=64
        new_rad_elo64, new_dir_elo64 = calculate_elo(rad_ratings_before['elo64'], dir_ratings_before['elo64'], radiant_win, k=64)
        # Glicko-2
        new_rad_glicko, new_dir_glicko = env.rate_1vs1(rad_ratings_before['glicko2'], dir_ratings_before['glicko2'], drawn=(radiant_win==0.5))

        # Update the ratings dictionary
        team_ratings[rad_id] = {'elo32': new_rad_elo32, 'elo64': new_rad_elo64, 'glicko2': new_rad_glicko}
        team_ratings[dir_id] = {'elo32': new_dir_elo32, 'elo64': new_dir_elo64, 'glicko2': new_dir_glicko}

    print("Finished processing.")
    return pd.DataFrame(training_data)

# Remember to include the calculate_elo helper function in your script
def calculate_elo(rating_a, rating_b, outcome_a, k=32):
    # ... (same function as before)
    expected_a = 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    new_rating_a = rating_a + k * (outcome_a - expected_a)
    outcome_b = 1 - outcome_a
    expected_b = 1 - expected_a
    new_rating_b = rating_b + k * (outcome_b - expected_b)
    return new_rating_a, new_rating_b
# --- Main Execution ---
if __name__ == "__main__":
    NUMBER_OF_TOURNAMENTS = 300

    # 1. Fetch the team ID -> name mapping first
    teams_map = get_teams_map(path='teams.json')
    # 2. Fetch all the match data
    all_matches = get_matches_from_leagues(num_leagues=NUMBER_OF_TOURNAMENTS)

    # 3. Process matches and create the dataset, passing the teams_map
    if all_matches and teams_map:
        # dataset_df = create_match_dataset(all_matches, teams_map)
        dataset_df = create_match_dataset_multi_rating(all_matches, teams_map)

        # output_path = 'dota2_leagues_with_names_and_elo.csv'
        output_path = 'dota2_multi_rating_dataset.csv'
        dataset_df.to_csv(output_path, index=False)

        print(f"\nDataset successfully created!")
        print(f"Saved {len(dataset_df)} match records to {output_path}")
        print("\nFirst 5 rows of the new dataset:")
        print(dataset_df.head())
