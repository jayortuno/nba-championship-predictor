import pandas as pd

from nba_api.stats.endpoints import commonallplayers

# Load unmatched players
unmatched_df = pd.read_csv(
    "player data/2026-27_unmatched_players.csv"
)

# Get NBA player season history
nba_players = commonallplayers.CommonAllPlayers(
    is_only_current_season=0,
    league_id="00",
    season="2026-27",
    timeout=30
).get_data_frames()[0]

# Keep the columns we need
nba_players = nba_players[
    ["PERSON_ID", "FROM_YEAR", "TO_YEAR"]
]

# Match using NBA player IDs
unmatched_df = unmatched_df.merge(
    nba_players,
    left_on="NBA_Player_ID",
    right_on="PERSON_ID",
    how="left"
)

# Convert first NBA season to a number
unmatched_df["FROM_YEAR"] = pd.to_numeric(
    unmatched_df["FROM_YEAR"],
    errors="coerce"
)

# Classify experience
def classify_experience(row):
    if pd.isna(row["FROM_YEAR"]):
        return "Unknown"
    elif row["FROM_YEAR"] < 2026:
        return "Previous NBA Experience"
    else:
        return "No Earlier NBA Season Listed"


unmatched_df["Experience_Status"] = unmatched_df.apply(
    classify_experience,
    axis=1
)

print()
print("==============================")
print("NBA EXPERIENCE RESULTS")
print("==============================")

print(
    unmatched_df["Experience_Status"].value_counts()
)

print()
print(
    unmatched_df[
        ["Player", "Team", "FROM_YEAR", "Experience_Status"]
    ].head(30).to_string(index=False)
)

# =====================================
# SAVE PLAYERS WITH NBA EXPERIENCE
# =====================================

experienced_players = unmatched_df[
    unmatched_df["Experience_Status"] == "Previous NBA Experience"
].copy()

experienced_players.to_csv(
    "player data/2026-27_experienced_unmatched.csv",
    index=False
)

print()
print("==============================")
print("EXPERIENCED PLAYERS")
print("==============================")

print(
    experienced_players[
        ["Player", "Team", "FROM_YEAR", "TO_YEAR"]
    ].to_string(index=False)
)

print()
print("Total Experienced Players:", len(experienced_players))

# =====================================
# CHECK POSSIBLE NAME MISMATCHES
# =====================================

from difflib import get_close_matches
import unicodedata


def normalize_name(name):
    name = str(name).lower().strip()

    name = unicodedata.normalize("NFKD", name)

    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )

    return name


historical_players = pd.read_csv(
    "player data/2025-26_player_final_teams.csv"
)

historical_players["Clean_Name"] = (
    historical_players["Player"].apply(normalize_name)
)

experienced_players["Clean_Name"] = (
    experienced_players["Player"].apply(normalize_name)
)

historical_names = historical_players["Clean_Name"].tolist()

print()
print("==============================")
print("POSSIBLE NAME MATCHES")
print("==============================")

for _, row in experienced_players.iterrows():

    matches = get_close_matches(
        row["Clean_Name"],
        historical_names,
        n=1,
        cutoff=0.65
    )

    if matches:
        matched_player = historical_players[
            historical_players["Clean_Name"] == matches[0]
        ]["Player"].iloc[0]

        print(
            row["Player"],
            " --> ",
            matched_player
        )
    else:
        print(row["Player"], " --> NO MATCH")