import pandas as pd
import unicodedata

# Load our historical player list
missing_players = pd.read_csv(
    "player data/2026-27_missing_player_history.csv"
)

# Load 2024-25 advanced statistics
advanced_df = pd.read_csv(
    "player data/2024-25_player_advanced.csv",
    encoding="cp1252"
)

# Clean the statistics
for column in ["BPM", "VORP", "MP"]:
    advanced_df[column] = pd.to_numeric(
        advanced_df[column],
        errors="coerce"
    )

# Remove repeated header rows
advanced_df = advanced_df.dropna(subset=["BPM", "VORP", "MP"])

# Normalize player names
def clean_name(name):
    name = str(name).lower().strip()
    name = unicodedata.normalize("NFKD", name)

    return "".join(
        char for char in name
        if not unicodedata.combining(char)
    )


missing_players["Clean_Name"] = (
    missing_players["Player"].apply(clean_name)
)

advanced_df["Clean_Name"] = (
    advanced_df["Player"].apply(clean_name)
)

# Prioritize total-season records for traded players
advanced_df["Total_Row"] = (
    advanced_df["Team"].astype(str)
    .str.match(r"^\d+TM$")
)

advanced_df = advanced_df.sort_values(
    "Total_Row",
    ascending=False
)

advanced_df = advanced_df.drop_duplicates(
    subset=["Clean_Name"],
    keep="first"
)

# Match the missing players
matched = missing_players.merge(
    advanced_df[["Clean_Name", "BPM", "VORP", "MP"]],
    on="Clean_Name",
    how="left",
    validate="one_to_one"
)

# Only use 2024-25 stats when that was the player's last season
matched.loc[
    matched["Last_Season"] != "2024-25",
    ["BPM", "VORP", "MP"]
] = float("nan")

# FIX CONFIRMED NAME MISMATCHES

name_fixes = {
    "Brandon Boston": "Brandon Boston Jr.",
    "Kevin Knox II": "Kevin Knox"
}

for roster_name, historical_name in name_fixes.items():

    player_stats = advanced_df[
        advanced_df["Player"] == historical_name
    ]

    if player_stats.empty:
        print("Not found:", historical_name)
        continue

    roster_mask = matched["Player"] == roster_name

    for column in ["BPM", "VORP", "MP"]:
        matched.loc[roster_mask, column] = (
            player_stats[column].iloc[0]
        )

    print("Recovered statistics for:", roster_name)

# Save the results
matched.to_csv(
    "player data/2026-27_recovered_advanced.csv",
    index=False
)

print()
print("==============================")
print("ADVANCED STATISTICS RECOVERY")
print("==============================")

print(
    matched[["Player", "Last_Season", "BPM", "VORP", "MP"]]
    .to_string(index=False)
)

print()
print(
    "Players With Advanced Stats:",
    matched["BPM"].notna().sum()
)

print(
    "Players Still Missing Advanced Stats:",
    matched["BPM"].isna().sum()
)

# CHECK REMAINING NAME MISMATCHES

from difflib import get_close_matches

print()
print("==============================")
print("POSSIBLE 2024-25 NAME MATCHES")
print("==============================")

for player_name in ["Brandon Boston", "Kevin Knox II"]:

    possible_matches = get_close_matches(
        clean_name(player_name),
        advanced_df["Clean_Name"].tolist(),
        n=5,
        cutoff=0.5
    )

    print()
    print(player_name)

    for match in possible_matches:

        original_name = advanced_df.loc[
            advanced_df["Clean_Name"] == match,
            "Player"
        ].iloc[0]

        print("  Possible match:", original_name)