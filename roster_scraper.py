import pandas as pd
import time

from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster

all_rosters = []

nba_teams = teams.get_teams()

for team in nba_teams:
    print("Getting roster:", team["full_name"])

    try:
        roster = commonteamroster.CommonTeamRoster(
            team_id=team["id"],
            season="2026-27",
            timeout=20
        ).get_data_frames()[0]

        roster["Team"] = team["abbreviation"]

        all_rosters.append(
            roster[["PLAYER", "PLAYER_ID", "Team"]]
        )

    except Exception as e:
        print("Error:", team["full_name"], e)

    time.sleep(1)

if all_rosters:
    roster_df = pd.concat(
        all_rosters,
        ignore_index=True
    )

    roster_df = roster_df.rename(
        columns={
            "PLAYER": "Player",
            "PLAYER_ID": "NBA_Player_ID"
        }
    )

    roster_df.to_csv(
        "player data/2026-27_nba_rosters.csv",
        index=False
    )

    print()
    print("Total Players:", len(roster_df))
    print("Teams Collected:", roster_df["Team"].nunique())
else:
    print("No roster data collected.")