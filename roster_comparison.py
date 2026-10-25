import pandas as pd

# Load datasets
old_rosters = pd.read_csv(
    "player data/2025-26_player_final_teams.csv"
)

new_rosters = pd.read_csv(
    "player data/2026-27_nba_rosters.csv"
)

# Match players by name
comparison = new_rosters.merge(
    old_rosters[
        ["Player", "Final_Team", "BPM", "VORP", "MP"]
    ],
    on="Player",
    how="left"
)

# Identify roster changes
def classify_player(row):
    if pd.isna(row["Final_Team"]):
        return "New Player"
    elif row["Team"] != row["Final_Team"]:
        return "Changed Teams"
    else:
        return "Stayed"


comparison["Status"] = comparison.apply(
    classify_player,
    axis=1
)

# Print results
print()
print("==============================")
print("ROSTER COMPARISON")
print("==============================")

print(comparison["Status"].value_counts())

print()
print("Players Who Changed Teams:")
print(
    comparison[
        comparison["Status"] == "Changed Teams"
    ][["Player", "Final_Team", "Team", "BPM", "VORP"]].head(20)
)

# Save comparison
comparison.to_csv(
    "player data/2026-27_roster_comparison.csv",
    index=False
)
# CHECK ROSTER DATA QUALITY


print()
print("==============================")
print("DATA QUALITY CHECK")
print("==============================")

print(
    "Duplicate NBA Player IDs:",
    new_rosters["NBA_Player_ID"].duplicated().sum()
)

print(
    "Duplicate Player Names:",
    new_rosters["Player"].duplicated().sum()
)

print()
print("UNMATCHED PLAYERS")
print("==============================")

unmatched = comparison[
    comparison["Status"] == "New Player"
]

print(
    unmatched[["Player", "Team"]].head(40).to_string(index=False)
)

import unicodedata

# IMPROVE PLAYER NAME MATCHING


def clean_name(name):
    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    name = unicodedata.normalize("NFKD", name)

    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )

    return name


old_rosters["Clean_Name"] = old_rosters["Player"].apply(
    clean_name
)

new_rosters["Clean_Name"] = new_rosters["Player"].apply(
    clean_name
)
# =====================================
# STANDARDIZE TEAM ABBREVIATIONS
# =====================================

new_rosters["Team"] = new_rosters["Team"].replace({
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO"
})

print("Team abbreviations standardized.")
# Check whether normalized names create duplicates
print()
print("Duplicate cleaned names (old):",
      old_rosters["Clean_Name"].duplicated().sum())

print("Duplicate cleaned names (new):",
      new_rosters["Clean_Name"].duplicated().sum())

# Match using cleaned names
improved_comparison = new_rosters.merge(
    old_rosters[
        ["Clean_Name", "Final_Team", "BPM", "VORP", "MP"]
    ],
    on="Clean_Name",
    how="left",
    validate="one_to_one"
)

# =====================================
# FIX CONFIRMED PLAYER NAME MATCHES
# =====================================

name_fixes = {
    "Karlo Matković": "Karlo Matkovi?",
    "DaRon Holmes II": "DaRon Holmes",
    "Nikola Jokić": "Nikola Joki?",
    "Kristaps Porziņģis": "Kristaps Porzi??is",
    "Jimmy Butler III": "Jimmy Butler",
    "Alperen Sengun": "Alperen ?engün",
    "Bogdan Bogdanović": "Bogdan Bogdanovi?",
    "Luka Dončić": "Luka Don?i?",
    "Nikola Jović": "Nikola Jovi?",
    "Bobby Portis Jr.": "Bobby Portis",
    "AJ Green": "A.J. Green",
    "Kasparas Jakučionis": "Kasparas Jaku?ionis",
    "Terrence Shannon Jr": "Terrence Shannon Jr.",
    "Nikola Vučević": "Nikola Vu?evi?",
    "Vít Krejčí": "Vít Krej?í",
    "Robert Williams III": "Robert Williams",
    "Nikola Topić": "Nikola Topi?",
    "Trey Jemison III": "Trey Jemison",
    "Jusuf Nurkić": "Jusuf Nurki?",
    "Walter Clayton Jr.": "Walter Clayton",
    "GG Jackson": "GG Jackson II",
    "Ronald Holland II": "Ron Holland"
}

# Only use names that exist in historical data
historical_lookup = historical_players = old_rosters.set_index("Player")

fixed_count = 0

for new_name, old_name in name_fixes.items():

    if old_name not in historical_lookup.index:
        print("Historical name not found:", old_name)
        continue

    mask = (
        (improved_comparison["Player"] == new_name)
        & (improved_comparison["Final_Team"].isna())
    )

    if mask.sum() != 1:
        continue

    historical_row = historical_lookup.loc[old_name]

    for column in ["Final_Team", "BPM", "VORP", "MP"]:
        improved_comparison.loc[mask, column] = historical_row[column]

    fixed_count += 1

print("Additional players matched:", fixed_count)
print(
    "Remaining unmatched:",
    improved_comparison["Final_Team"].isna().sum()
)

print()
print("==============================")
print("IMPROVED MATCHING RESULTS")
print("==============================")

print(
    "Matched Players:",
    improved_comparison["Final_Team"].notna().sum()
)

print(
    "Unmatched Players:",
    improved_comparison["Final_Team"].isna().sum()
)

# =====================================
# SAVE IMPROVED ROSTER COMPARISON
# =====================================

improved_comparison["Status"] = improved_comparison.apply(
    classify_player,
    axis=1
)

improved_comparison.to_csv(
    "player data/2026-27_roster_comparison.csv",
    index=False
)

print()
print("==============================")
print("UPDATED ROSTER COMPARISON")
print("==============================")

print(improved_comparison["Status"].value_counts())


unmatched_players = improved_comparison[
    improved_comparison["Status"] == "New Player"
].copy()

unmatched_players = unmatched_players[
    ["Player", "NBA_Player_ID", "Team"]
]

unmatched_players.to_csv(
    "player data/2026-27_unmatched_players.csv",
    index=False
)

print()
print("==============================")
print("UNMATCHED PLAYERS BY TEAM")
print("==============================")

print(
    unmatched_players["Team"]
    .value_counts()
    .sort_index()
)

print()
print("Total Unmatched:", len(unmatched_players))