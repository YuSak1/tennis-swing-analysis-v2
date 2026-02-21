import numpy as np
import pandas as pd
from pathlib import Path
from dtaidistance import dtw

from app.config import REFERENCES_DIR, FEATURE_GROUPS, PLAYERS


class ComparisonService:
    """Compares user swing to pro player references using Dynamic Time Warping."""

    def __init__(self):
        self.reference_data: dict[str, list[pd.DataFrame]] = {}

    def load(self):
        """Load pre-extracted reference feature CSVs for all players."""
        for player in PLAYERS:
            player_dir = REFERENCES_DIR / player
            if not player_dir.exists():
                continue

            self.reference_data[player] = []
            for csv_file in sorted(player_dir.glob("*.csv")):
                df = pd.read_csv(csv_file)
                self.reference_data[player].append(df)

        loaded = {p: len(refs) for p, refs in self.reference_data.items()}
        print(f"Loaded reference data: {loaded}")

    def compare(self, user_features_df: pd.DataFrame) -> dict:
        """
        Compare user swing to all pro player references.

        Args:
            user_features_df: DataFrame from FeatureService with position features
                              (velocity/acceleration columns are ignored here)

        Returns:
            {
                "Federer": {
                    "overall_similarity": 75.2,
                    "body_groups": {
                        "Racket Arm": {"similarity": 82.1, "per_feature": {...}},
                        ...
                    }
                },
                ...
            }
        """
        results = {}

        for player, ref_swings in self.reference_data.items():
            if not ref_swings:
                continue

            player_result = {"body_groups": {}}

            for group_name, feature_list in FEATURE_GROUPS.items():
                group_distances = {}

                for feature in feature_list:
                    if feature not in user_features_df.columns:
                        continue

                    user_seq = user_features_df[feature].values.astype(np.float64)

                    # Compare against all reference swings, take the best match
                    best_dist = float("inf")
                    for ref_df in ref_swings:
                        if feature not in ref_df.columns:
                            continue
                        ref_seq = ref_df[feature].values.astype(np.float64)

                        # Normalize both sequences (zero mean, unit variance)
                        user_norm = self._normalize(user_seq)
                        ref_norm = self._normalize(ref_seq)

                        dist = dtw.distance(user_norm, ref_norm, use_pruning=True)
                        best_dist = min(best_dist, dist)

                    group_distances[feature] = best_dist

                avg_distance = (
                    np.mean(list(group_distances.values()))
                    if group_distances
                    else float("inf")
                )

                player_result["body_groups"][group_name] = {
                    "distance": float(avg_distance),
                    "per_feature": {k: float(v) for k, v in group_distances.items()},
                }

            # Overall distance = mean across all body groups
            all_group_dists = [
                g["distance"] for g in player_result["body_groups"].values()
            ]
            player_result["overall_distance"] = float(np.mean(all_group_dists))
            results[player] = player_result

        # Convert distances → similarity percentages
        results = self._to_similarities(results)
        return results

    def _normalize(self, seq: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        std = seq.std()
        if std < 1e-8:
            return seq - seq.mean()
        return (seq - seq.mean()) / std

    def _to_similarities(self, results: dict) -> dict:
        """
        Convert DTW distances to 0-100 similarity scores.
        Uses softmax-style relative scoring so percentages sum meaningfully.
        """
        if not results:
            return results

        # Overall similarities
        overall_dists = {p: r["overall_distance"] for p, r in results.items()}
        overall_sims = self._distances_to_scores(overall_dists)

        for player in results:
            results[player]["overall_similarity"] = overall_sims[player]

            # Per body group similarities
            for group_name in results[player]["body_groups"]:
                group_dists = {
                    p: r["body_groups"][group_name]["distance"]
                    for p, r in results.items()
                    if group_name in r["body_groups"]
                }
                group_sims = self._distances_to_scores(group_dists)
                results[player]["body_groups"][group_name]["similarity"] = group_sims[
                    player
                ]

        return results

    def _distances_to_scores(self, distances: dict[str, float]) -> dict[str, float]:
        """
        Convert a dict of distances to similarity scores (0-100).
        Smaller distance = higher score. Uses negative exponential scaling.
        """
        if not distances:
            return {}

        values = np.array(list(distances.values()))
        # Negative exponential: e^(-d) gives higher scores for smaller distances
        exp_values = np.exp(-values / (np.median(values) + 1e-8))
        # Normalize to 0-100 range
        min_exp, max_exp = exp_values.min(), exp_values.max()
        if max_exp - min_exp > 1e-8:
            scores = (exp_values - min_exp) / (max_exp - min_exp) * 100
        else:
            scores = np.full_like(exp_values, 50.0)

        return dict(zip(distances.keys(), scores.tolist()))
