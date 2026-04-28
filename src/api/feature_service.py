"""
Live feature builder for the prediction API.

Given a (home_team_id, away_team_id, match_date) tuple, computes the same 
32 features the training pipeline produces, using the SQLite database as
the source of historical match and FIFA data.

This module mirrors the logic in src.features.build_features but operates 
on a single match instead of the full dataset.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


FIFA_NUMERIC_COLS = [
    'buildUpPlaySpeed', 'buildUpPlayDribbling', 'buildUpPlayPassing',
    'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting',
    'defencePressure', 'defenceAggression', 'defenceTeamWidth',
]

ROLLING_WINDOW = 5
H2H_WINDOW = 5


class FeatureService:
    """
    Stateful service that holds the SQLite connection and team lookup.
    Instantiated once at API startup, reused for every prediction request.
    """

    def __init__(self, db_path: Path, team_lookup_path: Path):
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"SQLite database not found at {db_path}. "
                f"Run `dvc pull` or `python -m src.features.build_features` first."
            )
        if not Path(team_lookup_path).exists():
            raise FileNotFoundError(
                f"Team lookup CSV not found at {team_lookup_path}. "
                f"Run `python -m src.features.build_features` first."
            )

        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.team_lookup = pd.read_csv(team_lookup_path)

        # Load ELO ratings (Day 7 experiment)
        team_elos_path = team_lookup_path.parent / "team_elos.csv"
        if team_elos_path.exists():
            team_elos_df = pd.read_csv(team_elos_path)
            self._team_elos = dict(zip(team_elos_df['team_api_id'], team_elos_df['elo']))
            logger.info(f"FeatureService loaded ELO ratings for {len(self._team_elos)} teams")
        else:
            logger.warning(f"team_elos.csv not found at {team_elos_path}; using ELO=1500 default")
            self._team_elos = {}

        # Compute FIFA column means for imputation (same approach as training)
        # We do this once at startup to match the train-time imputation
        self._fifa_means = self._compute_fifa_means()
        
        logger.info(f"FeatureService initialized with {len(self.team_lookup)} teams")

    def _compute_fifa_means(self) -> dict:
        """Compute column means from all FIFA snapshots (matches training behavior)."""
        query = f"SELECT {', '.join(FIFA_NUMERIC_COLS)} FROM Team_Attributes"
        df = pd.read_sql_query(query, self.conn)
        return df.mean().to_dict()

    def team_exists(self, team_api_id: int) -> bool:
        """Check if a team ID is known."""
        return team_api_id in self.team_lookup['team_api_id'].values

    def get_team_name(self, team_api_id: int) -> Optional[str]:
        """Resolve team ID to display name."""
        row = self.team_lookup[self.team_lookup['team_api_id'] == team_api_id]
        if len(row) == 0:
            return None
        return row.iloc[0]['team_long_name']

    def list_teams(self) -> list[dict]:
        """Return all teams for the frontend dropdown."""
        return [
            {
                'team_api_id': int(row['team_api_id']),
                'name': row['team_long_name'],
                'short_name': row['team_short_name'],
            }
            for _, row in self.team_lookup.iterrows()
        ]

    def build_features(
        self, home_team_id: int, away_team_id: int, match_date: datetime
    ) -> pd.DataFrame:
        """
        Build the 35-feature vector for a single match.

        Returns a DataFrame with one row, matching the column order the model expects.
        """
        logger.info(f"Building features for home={home_team_id}, away={away_team_id}, date={match_date.date()}")

        features = {}

        # 1. Rolling form for home and away
        home_form = self._compute_team_form(home_team_id, match_date)
        away_form = self._compute_team_form(away_team_id, match_date)
        for k, v in home_form.items():
            features[f'home_{k}'] = v
        for k, v in away_form.items():
            features[f'away_{k}'] = v

        # 2. Head-to-head
        h2h = self._compute_head_to_head(home_team_id, away_team_id, match_date)
        features.update(h2h)

        # 3. FIFA attributes for home and away
        home_fifa = self._get_fifa_snapshot(home_team_id, match_date)
        away_fifa = self._get_fifa_snapshot(away_team_id, match_date)
        for k, v in home_fifa.items():
            features[f'home_{k}'] = v
        for k, v in away_fifa.items():
            features[f'away_{k}'] = v
        
        # 4. ELO ratings (Day 7 experiment)
        home_elo = float(self._team_elos.get(home_team_id, 1500.0))
        away_elo = float(self._team_elos.get(away_team_id, 1500.0))
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo

        # Return as single-row DataFrame in the exact column order the model expects
        return pd.DataFrame([features])[self._expected_columns()]

    def _compute_team_form(self, team_id: int, match_date: datetime) -> dict:
        """Compute rolling form over the team's last ROLLING_WINDOW matches before match_date."""
        query = """
            SELECT date, home_team_api_id, away_team_api_id, 
                   home_team_goal, away_team_goal
            FROM Match
            WHERE (home_team_api_id = ? OR away_team_api_id = ?)
              AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """
        recent = pd.read_sql_query(
            query, self.conn,
            params=(team_id, team_id, match_date.strftime('%Y-%m-%d'), ROLLING_WINDOW),
        )

        if len(recent) < ROLLING_WINDOW:
            # Cold-start — return zeros (model trained on matches with full history,
            # but at inference we degrade gracefully rather than refuse the request)
            logger.warning(f"  Team {team_id} has only {len(recent)} prior matches, using zeros")
            return {
                'form_wins': 0, 'form_draws': 0, 'form_losses': 0,
                'form_gs_avg': 0.0, 'form_gc_avg': 0.0,
            }

        wins = draws = losses = 0
        gs_total = gc_total = 0
        for _, row in recent.iterrows():
            if row['home_team_api_id'] == team_id:
                gs, gc = row['home_team_goal'], row['away_team_goal']
            else:
                gs, gc = row['away_team_goal'], row['home_team_goal']
            gs_total += gs
            gc_total += gc
            if gs > gc:
                wins += 1
            elif gs < gc:
                losses += 1
            else:
                draws += 1

        return {
            'form_wins': wins,
            'form_draws': draws,
            'form_losses': losses,
            'form_gs_avg': gs_total / ROLLING_WINDOW,
            'form_gc_avg': gc_total / ROLLING_WINDOW,
        }

    def _compute_head_to_head(
        self, home_team_id: int, away_team_id: int, match_date: datetime
    ) -> dict:
        """Compute H2H features over last H2H_WINDOW meetings between these teams."""
        query = """
            SELECT date, home_team_api_id, away_team_api_id,
                   home_team_goal, away_team_goal
            FROM Match
            WHERE ((home_team_api_id = ? AND away_team_api_id = ?)
                OR (home_team_api_id = ? AND away_team_api_id = ?))
              AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """
        meetings = pd.read_sql_query(
            query, self.conn,
            params=(
                home_team_id, away_team_id,
                away_team_id, home_team_id,
                match_date.strftime('%Y-%m-%d'),
                H2H_WINDOW,
            ),
        )

        if len(meetings) == 0:
            return {
                'h2h_home_wins': 0, 'h2h_draws': 0,
                'h2h_away_wins': 0, 'h2h_n_meetings': 0,
            }

        # Count outcomes from the *current* home team's perspective
        h_wins = d = a_wins = 0
        for _, row in meetings.iterrows():
            if row['home_team_api_id'] == home_team_id:
                # In that past match, our current home was home; apply normal logic
                if row['home_team_goal'] > row['away_team_goal']:
                    h_wins += 1
                elif row['home_team_goal'] < row['away_team_goal']:
                    a_wins += 1
                else:
                    d += 1
            else:
                # In that past match, our current home was the away team; flip
                if row['home_team_goal'] > row['away_team_goal']:
                    a_wins += 1
                elif row['home_team_goal'] < row['away_team_goal']:
                    h_wins += 1
                else:
                    d += 1

        return {
            'h2h_home_wins': h_wins,
            'h2h_draws': d,
            'h2h_away_wins': a_wins,
            'h2h_n_meetings': len(meetings),
        }

    def _get_fifa_snapshot(self, team_id: int, match_date: datetime) -> dict:
        """Get most recent FIFA snapshot for team before match_date."""
        query = f"""
            SELECT {', '.join(FIFA_NUMERIC_COLS)}
            FROM Team_Attributes
            WHERE team_api_id = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
        """
        result = pd.read_sql_query(
            query, self.conn,
            params=(team_id, match_date.strftime('%Y-%m-%d')),
        )

        if len(result) == 0:
            # No snapshot before this date — use column means (matches training imputation)
            logger.warning(f"  No FIFA snapshot for team {team_id} before {match_date.date()}, using means")
            return {col: self._fifa_means[col] for col in FIFA_NUMERIC_COLS}

        row = result.iloc[0]
        return {
            col: (row[col] if pd.notna(row[col]) else self._fifa_means[col])
            for col in FIFA_NUMERIC_COLS
        }

    def _expected_columns(self) -> list[str]:
        """The exact column order the model expects (matches train.py FEATURE_COLUMNS)."""
        return [
            'home_form_wins', 'home_form_draws', 'home_form_losses',
            'home_form_gs_avg', 'home_form_gc_avg',
            'away_form_wins', 'away_form_draws', 'away_form_losses',
            'away_form_gs_avg', 'away_form_gc_avg',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_n_meetings',
            'home_buildUpPlaySpeed', 'home_buildUpPlayDribbling', 'home_buildUpPlayPassing',
            'home_chanceCreationPassing', 'home_chanceCreationCrossing', 'home_chanceCreationShooting',
            'home_defencePressure', 'home_defenceAggression', 'home_defenceTeamWidth',
            'away_buildUpPlaySpeed', 'away_buildUpPlayDribbling', 'away_buildUpPlayPassing',
            'away_chanceCreationPassing', 'away_chanceCreationCrossing', 'away_chanceCreationShooting',
            'away_defencePressure', 'away_defenceAggression', 'away_defenceTeamWidth',
            # ELO (3) — Day 7 experiment
            'home_elo', 'away_elo', 'elo_diff',
        ]