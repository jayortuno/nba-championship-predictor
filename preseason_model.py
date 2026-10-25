import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df= pd.read_csv("nba_historical_team_stats.csv")

df = df.sort_values(["Team", "Season"])

features = [
    "W", "MOV", "SRS", "NRtg",
    "ORtg", "DRtg",
    "eFG%", "TS%", "TOV%",
    "ORB%", "DRB%"
]

df["Target_Champion"] = df["Champion"]

df["Next_Champion"] = (
    df.groupby("Team")["Target_Champion"].shift(-1)
)

df["Next_Season"] = (
    df.groupby("Team")["Season"].shift(-1)
)

preseason_df = df.dropna(
    subset=["Next_Champion", "Next_Season"]
).copy()

preseason_df["Next_Champion"] = (
    preseason_df["Next_Champion"].astype(int)
)

preseason_df.to_csv(
    "nba_preseason_training_data.csv",
    index = False)

print()
print("==============================")
print("PRESEASON DATASET CREATED")
print("==============================")

print("Total Rows:", len(preseason_df))
print("Seasons Predicted:", preseason_df["Next_Season"].nunique())
print("Champions:", preseason_df["Next_Champion"].sum())

print()
print(
    preseason_df[
        ["Team", "Season", "Next_Season", "Next_Champion"]
    ].head(15).to_string(index=False)
)

results = []

seasons = sorted(preseason_df["Next_Season"].unique())

for i in range(1, len(seasons)):

    test_season = seasons[i]
    training_seasons = seasons[:i]

    train = preseason_df[
        preseason_df["Next_Season"].isin(training_seasons)
    ]

    test = preseason_df[
        preseason_df["Next_Season"] == test_season
    ].copy()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        train[features],
        train["Next_Champion"]
    )

    test["Score"] = model.predict_proba(
        test[features]
    )[:, 1]

    test = test.sort_values(
        "Score",
        ascending=False
    ).reset_index(drop=True)

    actual_champion = test.loc[
        test["Next_Champion"] == 1, "Team"
    ].iloc[0]

    predicted_champion = test.iloc[0]["Team"]

    champion_rank = (
        test.index[test["Next_Champion"] == 1][0] + 1
    )

    results.append({
        "Season": test_season,
        "Predicted": predicted_champion,
        "Actual": actual_champion,
        "Champion_Rank": champion_rank,
        "Correct": predicted_champion == actual_champion
    })

    print()
    print(f"SEASON: {test_season}")
    print(f"Predicted: {predicted_champion}")
    print(f"Actual: {actual_champion}")
    print(f"Champion Rank: {champion_rank}")

# =====================================
# MODEL PERFORMANCE
# =====================================

results_df = pd.DataFrame(results)

print()
print("==============================")
print("PRESEASON MODEL PERFORMANCE")
print("==============================")

print(
    "Top-1 Accuracy:",
    f"{results_df['Correct'].mean() * 100:.1f}%"
)

for n in [2, 3, 5]:

    accuracy = (
        results_df["Champion_Rank"] <= n
    ).mean() * 100

    print(f"Top-{n} Accuracy: {accuracy:.1f}%")

print(
    "Average Champion Rank:",
    round(results_df["Champion_Rank"].mean(), 2)
)

results_df.to_csv(
    "nba_preseason_model_results.csv",
    index=False
)

# PRESEASON PREDICTIONS

final_preseason_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

final_preseason_model.fit(
    preseason_df[features],
    preseason_df["Next_Champion"]
)

future_teams = pd.read_csv("nba_2026_27_roster_adjusted.csv")

future_teams["Preseason_Score"] = (
    final_preseason_model.predict_proba(
        future_teams[features]
    )[:, 1]
)

total_score = future_teams["Preseason_Score"].sum()

if total_score == 0:
    raise ValueError("All preseason model scores are zero.")

future_teams["Normalized_Preseason_Score"] = (
    future_teams["Preseason_Score"]
    / total_score
    * 100
)

future_teams = future_teams.sort_values(
    "Normalized_Preseason_Score",
    ascending=False
).reset_index(drop=True)

future_teams["Preseason_Rank"] = (
    future_teams.index + 1
)

print()
print("==============================")
print("2026-27 PRESEASON PREDICTIONS")
print("==============================")

print(
    future_teams[
        [
            "Preseason_Rank",
            "Team",
            "Normalized_Preseason_Score",
        ]
    ].round(2).to_string(index=False)
)

future_teams.to_csv(
    "nba_2026_27_preseason_predictions.csv",
    index=False
)