import pandas as pd
from sklearn.ensemble import RandomForestClassifier

historical = pd.read_csv(
    "nba_historical_team_stats.csv"
)

preseason = pd.read_csv(
    "nba_preseason_training_data.csv"
)

features = [
    "W", "MOV", "SRS", "NRtg",
    "ORtg", "DRtg",
    "eFG%", "TS%", "TOV%",
    "ORB%", "DRB%"
]

seasons = sorted(preseason["Next_Season"].unique())

results = []


def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )


# =====================================
# TEST BOTH MODELS
# =====================================

for i in range(1, len(seasons)):

    test_season = seasons[i]

    # ---------------------------------
    # MODEL 1: ORIGINAL APPROACH
    # ---------------------------------

    # Train on completed seasons before
    # the season we're predicting
    original_train = historical[
        historical["Season"] < test_season
    ]

    original_model = create_model()

    original_model.fit(
        original_train[features],
        original_train["Champion"]
    )

    # Use previous-season statistics
    original_test = preseason[
        preseason["Next_Season"] == test_season
    ].copy()

    original_test["Score"] = (
        original_model.predict_proba(
            original_test[features]
        )[:, 1]
    )

    original_test = original_test.sort_values(
        "Score",
        ascending=False
    ).reset_index(drop=True)

    original_rank = (
        original_test.index[
            original_test["Next_Champion"] == 1
        ][0] + 1
    )

    original_pick = original_test.iloc[0]["Team"]

    # ---------------------------------
    # MODEL 2: PRESEASON APPROACH
    # ---------------------------------

    preseason_train = preseason[
        preseason["Next_Season"] < test_season
    ]

    preseason_model = create_model()

    preseason_model.fit(
        preseason_train[features],
        preseason_train["Next_Champion"]
    )

    preseason_test = preseason[
        preseason["Next_Season"] == test_season
    ].copy()

    preseason_test["Score"] = (
        preseason_model.predict_proba(
            preseason_test[features]
        )[:, 1]
    )

    preseason_test = preseason_test.sort_values(
        "Score",
        ascending=False
    ).reset_index(drop=True)

    preseason_rank = (
        preseason_test.index[
            preseason_test["Next_Champion"] == 1
        ][0] + 1
    )

    preseason_pick = preseason_test.iloc[0]["Team"]

    actual_champion = preseason_test.loc[
        preseason_test["Next_Champion"] == 1,
        "Team"
    ].iloc[0]

    results.append({
        "Season": test_season,
        "Actual": actual_champion,
        "Original_Pick": original_pick,
        "Original_Rank": original_rank,
        "Preseason_Pick": preseason_pick,
        "Preseason_Rank": preseason_rank
    })

    print()
    print("==============================")
    print(f"SEASON: {test_season}")
    print("==============================")

    print("Actual Champion:", actual_champion)
    print(
        f"Original Model: {original_pick} "
        f"(Champion Rank: {original_rank})"
    )
    print(
        f"Preseason Model: {preseason_pick} "
        f"(Champion Rank: {preseason_rank})"
    )


# =====================================
# COMPARE PERFORMANCE
# =====================================

results_df = pd.DataFrame(results)

print()
print("==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

for name, column in [
    ("Original Model", "Original_Rank"),
    ("Preseason Model", "Preseason_Rank")
]:

    ranks = results_df[column]

    print()
    print(name)
    print("------------------------------")

    for n in [1, 2, 3, 5]:
        accuracy = (ranks <= n).mean() * 100
        print(f"Top-{n} Accuracy: {accuracy:.1f}%")

    print(
        "Average Champion Rank:",
        round(ranks.mean(), 2)
    )


results_df.to_csv(
    "nba_fair_model_comparison.csv",
    index=False
)

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

# Load our 2026-27 roster-adjusted dataset
future_teams = pd.read_csv(
    "nba_2026_27_roster_adjusted.csv"
)

# Generate championship scores
future_teams["Preseason_Score"] = (
    final_preseason_model.predict_proba(
        future_teams[features]
    )[:, 1]
)

# Normalize scores across all 30 teams
total_score = future_teams["Preseason_Score"].sum()

if total_score == 0:
    raise ValueError("All preseason model scores are zero.")

future_teams["Normalized_Preseason_Score"] = (
    future_teams["Preseason_Score"]
    / total_score
    * 100
)

# Sort teams from highest to lowest
future_teams = future_teams.sort_values(
    "Normalized_Preseason_Score",
    ascending=False
).reset_index(drop=True)

# Add rankings
future_teams["Preseason_Rank"] = (
    future_teams.index + 1
)

# Display results
print()
print("==============================")
print("2026-27 PRESEASON PREDICTIONS")
print("==============================")

print(
    future_teams[
        [
            "Preseason_Rank",
            "Team",
            "Normalized_Preseason_Score"
        ]
    ].round(2).to_string(index=False)
)

# Save results
future_teams.to_csv(
    "nba_2026_27_preseason_predictions.csv",
    index=False
)