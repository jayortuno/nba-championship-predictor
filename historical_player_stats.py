import pandas as pd
import time

from nba_api.stats.endpoints import playercareerstats

# Load experienced players
players_df = pd.read_csv(
    "player data/2026-27_experienced_unmatched.csv"
)

historical_stats = []

for _, player in players_df.iterrows():

    player_name = player["Player"]
    player_id = int(player["NBA_Player_ID"])

    print("Checking:", player_name)

    try:
        career = playercareerstats.PlayerCareerStats(
            player_id=player_id,
            timeout=30
        )

        seasons = career.get_data_frames()[0]

        # Keep only regular-season totals
        seasons = seasons[
            seasons["SEASON_ID"] < "2026-27"
        ]

        # Exclude career total rows
        seasons = seasons[
            seasons["SEASON_ID"].notna()
        ]

        if not seasons.empty:

            latest_season = seasons.iloc[-1]

            historical_stats.append({
                "Player": player_name,
                "NBA_Player_ID": player_id,
                "Team": player["Team"],
                "Last_Season": latest_season["SEASON_ID"],
                "Games": latest_season["GP"],
                "Minutes": latest_season["MIN"],
                "Points": latest_season["PTS"],
                "Rebounds": latest_season["REB"],
                "Assists": latest_season["AST"]
            })

        else:
            print("No previous statistics:", player_name)

    except Exception as e:
        print("Error retrieving:", player_name, e)

    time.sleep(1)

# Save results
historical_df = pd.DataFrame(historical_stats)

historical_df.to_csv(
    "player data/2026-27_missing_player_history.csv",
    index=False
)

print()
print("==============================")
print("HISTORICAL PLAYER RESULTS")
print("==============================")

print(historical_df.to_string(index=False))

print()
print("Players With Statistics:", len(historical_df))