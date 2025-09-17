import json
import requests
import time

BASE_URL = "https://api.opendota.com/api"

if __name__ == "__main__":
    """
    Fetches all teams from OpenDota using pagination and returns a mapping of team_id to name.
    """
    print("Fetching all team data with pagination...")
    all_teams_data = []
    page_num = 0

    while True:
        try:
            print(f"  Fetching page {page_num}...")
            response = requests.get(f"{BASE_URL}/teams?page={page_num}")
            response.raise_for_status()

            page_data = response.json()

            # If the page returns no data, we've reached the end
            if not page_data:
                print("  No more teams found. Finished fetching.")
                break

            all_teams_data.extend(page_data)
            page_num += 1

            # Be respectful of the API rate limit, even for quick calls
            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            print(f"  An error occurred while fetching page {page_num}: {e}")
            break

    # Create the dictionary map from the aggregated list
    teams_map = {team['team_id']: team['name'] for team in all_teams_data if 'team_id' in team and 'name' in team}
    print(f"Successfully mapped {len(teams_map)} teams from {page_num} pages.")
    with open("teams.json", "w") as f:
        json.dump(teams_map, f, indent=4)
        print("Successfully saved 'teams.json'.")
