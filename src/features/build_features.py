"""
Feature engineering pipeline for the football-mlops project.

Loads raw match, team, and team-attribute data from SQLite, derives the 
target variable (H/D/A), engineers rolling form, head-to-head, and FIFA 
features, and writes train/test/production CSV splits.

Usage:
    # Run with defaults (paths from config.yaml, params from params.yaml):
    python -m src.features.build_features
    
    # Override params via CLI:
    python -m src.features.build_features --rolling-window 10 --h2h-window 5
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class encoding for LightGBM. Order matches the natural football display:
# Home Win | Draw | Away Win
OUTCOME_TO_INT = {'H': 0, 'D': 1, 'A': 2}
INT_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_INT.items()}

# The 9 numeric FIFA columns we use as features
FIFA_NUMERIC_COLS = [
    'buildUpPlaySpeed', 'buildUpPlayDribbling', 'buildUpPlayPassing',
    'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting',
    'defencePressure', 'defenceAggression', 'defenceTeamWidth',
]


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def load_raw_data(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load match, team, and team-attribute data from SQLite."""
    logger.info(f"Loading raw data from {db_path}")
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. "
            f"Run `dvc pull` to fetch the raw data, or check config.yaml's data.raw_db path."
        )

    conn = sqlite3.connect(db_path)

    matches_query = """
        SELECT 
            id, country_id, league_id, season, stage, date,
            home_team_api_id, away_team_api_id,
            home_team_goal, away_team_goal
        FROM Match
        ORDER BY date
    """
    matches = pd.read_sql_query(matches_query, conn)
    matches['date'] = pd.to_datetime(matches['date'])
    logger.info(f"  Loaded {len(matches):,} matches")

    teams_query = """
        SELECT team_api_id, team_fifa_api_id, team_long_name, team_short_name
        FROM Team
    """
    teams = pd.read_sql_query(teams_query, conn)
    logger.info(f"  Loaded {len(teams):,} teams")

    fifa_cols_sql = ", ".join(FIFA_NUMERIC_COLS)
    team_attrs_query = f"""
        SELECT 
            team_api_id, team_fifa_api_id, date,
            {fifa_cols_sql}
        FROM Team_Attributes
        ORDER BY team_api_id, date
    """
    team_attrs = pd.read_sql_query(team_attrs_query, conn)
    team_attrs['date'] = pd.to_datetime(team_attrs['date'])
    logger.info(f"  Loaded {len(team_attrs):,} team attribute snapshots")

    conn.close()
    return matches, teams, team_attrs


def derive_target(matches: pd.DataFrame) -> pd.DataFrame:
    """Add H/D/A outcome and integer-encoded outcome columns."""
    logger.info("Deriving target variable (H/D/A)")
    
    conditions = [
        matches['home_team_goal'] > matches['away_team_goal'],
        matches['home_team_goal'] < matches['away_team_goal'],
    ]
    choices = ['H', 'A']
    matches = matches.copy()
    matches['outcome'] = np.select(conditions, choices, default='D')
    matches['outcome_encoded'] = matches['outcome'].map(OUTCOME_TO_INT)

    null_count = matches['outcome'].isna().sum()
    if null_count > 0:
        logger.error(f"Found {null_count} matches with null outcomes!")
        raise ValueError(f"Target derivation produced {null_count} nulls — investigate")

    dist = matches['outcome'].value_counts(normalize=True).round(3).to_dict()
    logger.info(f"  Outcome distribution: {dist}")
    return matches


def _build_team_centric_view(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert match-centric data (1 row per match) to team-centric (2 rows per match)."""
    home_view = pd.DataFrame({
        'match_id': matches['id'],
        'date': matches['date'],
        'team_api_id': matches['home_team_api_id'],
        'is_home': True,
        'goals_scored': matches['home_team_goal'],
        'goals_conceded': matches['away_team_goal'],
    })
    home_view['won'] = (home_view['goals_scored'] > home_view['goals_conceded']).astype(int)
    home_view['drew'] = (home_view['goals_scored'] == home_view['goals_conceded']).astype(int)
    home_view['lost'] = (home_view['goals_scored'] < home_view['goals_conceded']).astype(int)

    away_view = pd.DataFrame({
        'match_id': matches['id'],
        'date': matches['date'],
        'team_api_id': matches['away_team_api_id'],
        'is_home': False,
        'goals_scored': matches['away_team_goal'],
        'goals_conceded': matches['home_team_goal'],
    })
    away_view['won'] = (away_view['goals_scored'] > away_view['goals_conceded']).astype(int)
    away_view['drew'] = (away_view['goals_scored'] == away_view['goals_conceded']).astype(int)
    away_view['lost'] = (away_view['goals_scored'] < away_view['goals_conceded']).astype(int)

    team_centric = pd.concat([home_view, away_view], ignore_index=True)
    team_centric = team_centric.sort_values(['team_api_id', 'date']).reset_index(drop=True)
    return team_centric


def compute_rolling_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling-form features (10 columns) for both teams."""
    logger.info(f"Computing rolling form features (window={window})")

    team_centric = _build_team_centric_view(matches)
    logger.info(f"  Team-centric view: {len(team_centric):,} rows")

    grouped = team_centric.groupby('team_api_id', group_keys=False)
    
    team_centric['form_wins'] = grouped['won'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=window).sum()
    )
    team_centric['form_draws'] = grouped['drew'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=window).sum()
    )
    team_centric['form_losses'] = grouped['lost'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=window).sum()
    )
    team_centric['form_gs_avg'] = grouped['goals_scored'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=window).mean()
    )
    team_centric['form_gc_avg'] = grouped['goals_conceded'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=window).mean()
    )

    home_stats = team_centric[team_centric['is_home']].set_index('match_id')[
        ['form_wins', 'form_draws', 'form_losses', 'form_gs_avg', 'form_gc_avg']
    ].add_prefix('home_')
    away_stats = team_centric[~team_centric['is_home']].set_index('match_id')[
        ['form_wins', 'form_draws', 'form_losses', 'form_gs_avg', 'form_gc_avg']
    ].add_prefix('away_')

    matches = matches.copy()
    matches = matches.merge(home_stats, left_on='id', right_index=True, how='left')
    matches = matches.merge(away_stats, left_on='id', right_index=True, how='left')

    initial_count = len(matches)
    form_cols = [c for c in matches.columns 
                 if c.endswith(('_wins', '_draws', '_losses', '_gs_avg', '_gc_avg'))]
    matches = matches.dropna(subset=form_cols)
    dropped = initial_count - len(matches)
    logger.info(f"  Dropped {dropped:,} matches without full form history (cold-start)")
    logger.info(f"  Remaining: {len(matches):,} matches with form features")
    return matches


def compute_head_to_head(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add head-to-head features (4 columns) for each match pair."""
    logger.info(f"Computing head-to-head features (window={window})")

    matches = matches.copy()
    matches['team_pair'] = matches.apply(
        lambda r: tuple(sorted([r['home_team_api_id'], r['away_team_api_id']])),
        axis=1
    )
    matches['curr_home_won'] = (matches['home_team_goal'] > matches['away_team_goal']).astype(int)
    matches['curr_drew'] = (matches['home_team_goal'] == matches['away_team_goal']).astype(int)
    matches['curr_away_won'] = (matches['home_team_goal'] < matches['away_team_goal']).astype(int)

    matches = matches.sort_values(['team_pair', 'date']).reset_index(drop=True)

    matches['lower_id'] = matches['team_pair'].apply(lambda p: p[0])
    matches['lower_id_won'] = np.where(
        matches['home_team_api_id'] == matches['lower_id'],
        matches['curr_home_won'],
        matches['curr_away_won']
    )
    matches['higher_id_won'] = np.where(
        matches['home_team_api_id'] == matches['lower_id'],
        matches['curr_away_won'],
        matches['curr_home_won']
    )

    grouped = matches.groupby('team_pair', group_keys=False)
    matches['h2h_lower_wins_raw'] = grouped['lower_id_won'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    matches['h2h_higher_wins_raw'] = grouped['higher_id_won'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    matches['h2h_draws'] = grouped['curr_drew'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )
    matches['h2h_n_meetings'] = grouped['curr_drew'].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).count()
    )

    matches['h2h_home_wins'] = np.where(
        matches['home_team_api_id'] == matches['lower_id'],
        matches['h2h_lower_wins_raw'],
        matches['h2h_higher_wins_raw']
    )
    matches['h2h_away_wins'] = np.where(
        matches['home_team_api_id'] == matches['lower_id'],
        matches['h2h_higher_wins_raw'],
        matches['h2h_lower_wins_raw']
    )

    h2h_cols = ['h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_n_meetings']
    matches[h2h_cols] = matches[h2h_cols].fillna(0)

    scratch_cols = ['team_pair', 'curr_home_won', 'curr_drew', 'curr_away_won',
                    'lower_id', 'lower_id_won', 'higher_id_won',
                    'h2h_lower_wins_raw', 'h2h_higher_wins_raw']
    matches = matches.drop(columns=scratch_cols)
    matches = matches.sort_values('date').reset_index(drop=True)

    n_no_history = (matches['h2h_n_meetings'] == 0).sum()
    logger.info(f"  Matches with zero prior H2H meetings: {n_no_history:,} "
                f"({100*n_no_history/len(matches):.1f}%)")
    logger.info(f"  Mean prior H2H meetings: {matches['h2h_n_meetings'].mean():.2f}")
    logger.info(f"  Computed H2H features for {len(matches):,} matches")
    return matches


def attach_fifa_features(matches: pd.DataFrame, team_attrs: pd.DataFrame) -> pd.DataFrame:
    """Add 18 FIFA team attribute features using time-aware (backward) joins."""
    logger.info("Attaching FIFA team attribute features (time-aware join)")

    matches = matches.copy()
    attrs_slim = team_attrs[['team_api_id', 'date'] + FIFA_NUMERIC_COLS].copy()
    attrs_slim = attrs_slim.sort_values('date').reset_index(drop=True)
    matches = matches.sort_values('date').reset_index(drop=True)

    # Home team join
    home_attrs = attrs_slim.rename(columns={'team_api_id': 'home_team_api_id'})
    matches = pd.merge_asof(
        left=matches, right=home_attrs,
        on='date', by='home_team_api_id', direction='backward',
    )
    matches = matches.rename(columns={c: f'home_{c}' for c in FIFA_NUMERIC_COLS})

    # Away team join
    away_attrs = attrs_slim.rename(columns={'team_api_id': 'away_team_api_id'})
    matches = pd.merge_asof(
        left=matches, right=away_attrs,
        on='date', by='away_team_api_id', direction='backward',
    )
    matches = matches.rename(columns={c: f'away_{c}' for c in FIFA_NUMERIC_COLS})

    fifa_feature_cols = (
        [f'home_{c}' for c in FIFA_NUMERIC_COLS] +
        [f'away_{c}' for c in FIFA_NUMERIC_COLS]
    )

    n_missing_before = matches[fifa_feature_cols].isna().sum().sum()
    logger.info(f"  Missing FIFA values before imputation: {n_missing_before:,}")

    for col in fifa_feature_cols:
        if matches[col].isna().any():
            mean_val = matches[col].mean()
            matches[col] = matches[col].fillna(mean_val)

    n_missing_after = matches[fifa_feature_cols].isna().sum().sum()
    logger.info(f"  Missing FIFA values after imputation: {n_missing_after:,}")
    logger.info(f"  Added {len(fifa_feature_cols)} FIFA features")
    return matches

# ---------------------------------------------------------------------------
# ELO ratings
# ---------------------------------------------------------------------------

ELO_INITIAL = 1500
ELO_K_FACTOR = 20         # FIFA standard
ELO_HOME_ADVANTAGE = 60   # well-known empirical value for football


def compute_elo_ratings(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Walk through matches chronologically, updating ELO ratings.
    Adds home_elo, away_elo, elo_diff as PRE-match ratings (no leakage).
    Returns (matches_with_elo, final_team_elos).
    """
    logger.info("Computing ELO ratings (Day 7 experiment)")

    matches = matches.sort_values('date').reset_index(drop=True).copy()
    elos: dict[int, float] = {}

    home_elo_col = []
    away_elo_col = []

    for _, row in matches.iterrows():
        home_id = row['home_team_api_id']
        away_id = row['away_team_api_id']

        home_elo = elos.get(home_id, ELO_INITIAL)
        away_elo = elos.get(away_id, ELO_INITIAL)

        # Record PRE-match ratings (this is what the model sees — no leakage)
        home_elo_col.append(home_elo)
        away_elo_col.append(away_elo)

        # Compute expected outcome with home advantage
        adjusted_home = home_elo + ELO_HOME_ADVANTAGE
        expected_home = 1 / (1 + 10 ** ((away_elo - adjusted_home) / 400))
        expected_away = 1 - expected_home

        # Actual outcome (1.0 win, 0.5 draw, 0.0 loss)
        if row['home_team_goal'] > row['away_team_goal']:
            actual_home, actual_away = 1.0, 0.0
        elif row['home_team_goal'] < row['away_team_goal']:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        # Update ratings AFTER recording features
        elos[home_id] = home_elo + ELO_K_FACTOR * (actual_home - expected_home)
        elos[away_id] = away_elo + ELO_K_FACTOR * (actual_away - expected_away)

    matches['home_elo'] = home_elo_col
    matches['away_elo'] = away_elo_col
    matches['elo_diff'] = matches['home_elo'] - matches['away_elo']

    if elos:
        logger.info(f"  ELO range: [{min(elos.values()):.0f}, {max(elos.values()):.0f}]")
    logger.info(f"  Tracked {len(elos)} teams")
    return matches, elos


def chronological_split(
    matches: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    production_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split matches by season into train/test/production sets."""
    logger.info("Splitting matches chronologically by season")
    logger.info(f"  Train seasons:      {train_seasons}")
    logger.info(f"  Test seasons:       {test_seasons}")
    logger.info(f"  Production seasons: {production_seasons}")

    available = set(matches['season'].unique())
    requested = set(train_seasons + test_seasons + production_seasons)
    missing = requested - available
    if missing:
        raise ValueError(f"Seasons not found in data: {missing}")

    all_seasons = train_seasons + test_seasons + production_seasons
    if len(all_seasons) != len(set(all_seasons)):
        raise ValueError("A season appears in multiple splits — would cause leakage")

    train_df = matches[matches['season'].isin(train_seasons)].copy()
    test_df = matches[matches['season'].isin(test_seasons)].copy()
    production_df = matches[matches['season'].isin(production_seasons)].copy()

    for name, df in [('Train', train_df), ('Test', test_df), ('Production', production_df)]:
        dist = df['outcome'].value_counts(normalize=True).round(3).to_dict()
        logger.info(f"  {name:12s}: {len(df):,} matches  |  outcome: {dist}")

    return train_df, test_df, production_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Find project root by walking up to where config.yaml lives."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no config.yaml)")


def main(rolling_window: Optional[int] = None, h2h_window: Optional[int] = None):
    """
    Run the feature engineering pipeline end-to-end.
    
    Args:
        rolling_window: Override params.yaml's rolling_window_size.
        h2h_window: Override params.yaml's h2h_window_size.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE — START")
    logger.info("=" * 60)

    project_root = _project_root()
    with open(project_root / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(project_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    db_path = project_root / config['data']['raw_db']
    processed_dir = project_root / config['data']['processed_dir']
    production_dir = project_root / config['data']['production_dir']
    processed_dir.mkdir(parents=True, exist_ok=True)
    production_dir.mkdir(parents=True, exist_ok=True)

    rw = rolling_window if rolling_window is not None else params['features']['rolling_window_size']
    h2hw = h2h_window if h2h_window is not None else params['features']['h2h_window_size']

    matches, teams, team_attrs = load_raw_data(db_path)
    matches = derive_target(matches)
    matches = compute_rolling_form(matches, window=rw)
    matches = compute_head_to_head(matches, window=h2hw)
    matches = attach_fifa_features(matches, team_attrs)
    matches, final_elos = compute_elo_ratings(matches)

    train_df, test_df, production_df = chronological_split(
        matches,
        train_seasons=params['split']['train_seasons'],
        test_seasons=params['split']['test_seasons'],
        production_seasons=params['split']['production_seasons'],
    )

    # Save outputs
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    production_path = production_dir / "season_2015_16.csv"
    team_lookup_path = processed_dir / "team_lookup.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    production_df.to_csv(production_path, index=False)

    teams_clean = teams[['team_api_id', 'team_long_name', 'team_short_name']].copy()
    teams_clean = teams_clean.dropna(subset=['team_long_name']).sort_values('team_long_name')
    teams_clean.to_csv(team_lookup_path, index=False)

    # Save final ELO ratings for runtime serving (Day 7 experiment)
    team_elos_path = processed_dir / "team_elos.csv"
    team_elos_df = pd.DataFrame([
        {'team_api_id': tid, 'elo': rating}
        for tid, rating in final_elos.items()
    ])
    team_elos_df.to_csv(team_elos_path, index=False)
    logger.info(f"Saved team ELOs:      {team_elos_path} ({len(team_elos_df):,} teams)")

    logger.info(f"Saved train CSV:      {train_path} ({len(train_df):,} rows)")
    logger.info(f"Saved test CSV:       {test_path} ({len(test_df):,} rows)")
    logger.info(f"Saved production CSV: {production_path} ({len(production_df):,} rows)")
    logger.info(f"Saved team lookup:    {team_lookup_path} ({len(teams_clean):,} teams)")

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE — COMPLETE")
    logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument('--rolling-window', type=int, default=None,
                        help="Override rolling form window size from params.yaml")
    parser.add_argument('--h2h-window', type=int, default=None,
                        help="Override H2H window size from params.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(rolling_window=args.rolling_window, h2h_window=args.h2h_window)