import numpy as np
import pandas as pd

from app.config import FEATURE_GROUPS


# Coaching knowledge base: feature-level tips
COACHING_KNOWLEDGE = {
    "r_elbow_angle": {
        "name": "Hitting arm elbow angle",
        "good": "Your hitting arm elbow angle is very close to {player}'s technique.",
        "tip_high": (
            "Your hitting arm is quite straight at contact. "
            "A slight bend (120–140°) allows more wrist snap and spin potential."
        ),
        "tip_low": (
            "Your elbow is more bent than the pros. "
            "Try extending more through the contact zone for better power transfer."
        ),
    },
    "l_elbow_angle": {
        "name": "Non-racket arm elbow angle",
        "good": "Your non-racket arm mirrors {player}'s form well.",
        "tip_high": (
            "Try pulling your non-racket arm across your body more during the swing — "
            "this helps with torso rotation and balance."
        ),
        "tip_low": (
            "Your non-racket arm is tucked in tight. "
            "Using it for balance by extending outward can improve stability."
        ),
    },
    "r_shoulder_angle": {
        "name": "Racket arm shoulder rotation",
        "good": "Your shoulder rotation on the racket side is on point — similar to {player}.",
        "tip_high": (
            "Your racket arm shoulder opens up wide. "
            "This can reduce control — try keeping the arm closer to your body during the backswing."
        ),
        "tip_low": (
            "Your shoulder rotation is limited. "
            "A bigger shoulder turn during preparation creates more elastic energy for power."
        ),
    },
    "r_knee_angle": {
        "name": "Dominant side knee bend",
        "good": "Great knee bend — your leg drive is similar to {player}'s.",
        "tip_high": (
            "Your knees are quite straight. "
            "Bending your knees 10–15° more during preparation gives you a lower center of gravity "
            "and more upward force for topspin."
        ),
        "tip_low": (
            "Very deep knee bend! This is powerful but make sure you're not losing "
            "balance during the follow-through."
        ),
    },
    "l_knee_angle": {
        "name": "Support leg knee bend",
        "good": "Your support leg positioning is solid.",
        "tip_high": (
            "Your support leg could use more bend — "
            "this improves weight transfer from back foot to front foot."
        ),
        "tip_low": "Good support leg bend for stability.",
    },
    "torso_rotation": {
        "name": "Torso rotation",
        "good": "Excellent torso rotation — very similar to {player}'s coiling action.",
        "tip_high": (
            "You're over-rotating your shoulders relative to your hips. "
            "This can cause timing issues — try syncing the shoulder turn with the backswing."
        ),
        "tip_low": (
            "Your shoulder turn is limited compared to the pros. "
            "A bigger separation between hip and shoulder lines during preparation "
            "creates more rotational energy."
        ),
    },
    "stance_width": {
        "name": "Stance width",
        "good": "Your stance width is well-balanced, similar to {player}.",
        "tip_high": (
            "Your stance is quite wide — make sure this isn't limiting "
            "your ability to transfer weight forward."
        ),
        "tip_low": (
            "A wider stance (shoulder-width or more) improves your base stability "
            "and helps with weight transfer."
        ),
    },
    "r_wrist_height": {
        "name": "Racket hand height",
        "good": "Your racket hand path looks great — tracking {player}'s pattern.",
        "tip_high": (
            "Your racket hand goes quite high in the backswing. "
            "A more compact takeback can improve consistency."
        ),
        "tip_low": (
            "Your racket hand could reach higher during the backswing — "
            "this creates a longer swing path for more racket head speed."
        ),
    },
    "body_lean": {
        "name": "Body lean",
        "good": "Your body balance is stable and similar to {player}.",
        "tip_high": (
            "You're leaning too far to one side during the swing. "
            "Try to stay more centered for better recovery."
        ),
        "tip_low": "Good lateral balance throughout the swing.",
    },
    "r_arm_extension": {
        "name": "Hitting arm extension",
        "good": "Great arm extension — you're reaching out like {player}.",
        "tip_high": (
            "You're reaching very far from your body at contact. "
            "Make sure you're hitting in your optimal contact zone."
        ),
        "tip_low": (
            "Try extending your arm more at contact — "
            "hitting too close to your body limits power."
        ),
    },
}


class FeedbackService:
    """Generates human-readable coaching feedback from comparison results."""

    def generate(
        self,
        comparison_results: dict,
        user_features_df: pd.DataFrame,
    ) -> list[dict]:
        """
        Generate coaching feedback based on DTW comparison results.

        Returns:
            list of {"type": "strength"|"improvement", "body_part": str, "message": str}
        """
        if not comparison_results:
            return []

        # Find most similar player
        best_player = max(
            comparison_results.items(),
            key=lambda x: x[1].get("overall_similarity", 0),
        )[0]

        tips = []

        for group_name, feature_list in FEATURE_GROUPS.items():
            for feature in feature_list:
                if feature not in COACHING_KNOWLEDGE:
                    continue

                knowledge = COACHING_KNOWLEDGE[feature]
                best_player_group = comparison_results[best_player]["body_groups"]

                if group_name not in best_player_group:
                    continue

                per_feature = best_player_group[group_name].get("per_feature", {})
                distance = per_feature.get(feature, float("inf"))

                if distance < 3.0:
                    # Strength — very similar to pro
                    tips.append(
                        {
                            "type": "strength",
                            "body_part": knowledge["name"],
                            "message": knowledge["good"].format(player=best_player),
                        }
                    )
                elif distance > 6.0:
                    # Needs improvement — determine direction
                    direction = self._get_direction(feature, user_features_df, comparison_results)
                    tip_key = "tip_high" if direction == "high" else "tip_low"
                    message = knowledge.get(tip_key, knowledge.get("tip_high", ""))
                    if message:
                        tips.append(
                            {
                                "type": "improvement",
                                "body_part": knowledge["name"],
                                "message": message,
                            }
                        )

        # Sort: improvements first, then strengths
        tips.sort(key=lambda x: 0 if x["type"] == "improvement" else 1)

        return tips

    def _get_direction(
        self,
        feature: str,
        user_features_df: pd.DataFrame,
        comparison_results: dict,
    ) -> str:
        """
        Determine if user's feature value is higher or lower than the pros' average.
        Returns 'high' or 'low'.
        """
        if feature not in user_features_df.columns:
            return "high"

        user_mean = user_features_df[feature].mean()

        # Calculate rough average across all pro references
        # (we only have distances, not raw values, so use a heuristic)
        # For a more precise comparison, we'd need to store reference means
        # For now, use the median as a reasonable proxy
        user_median = user_features_df[feature].median()

        # If the mean is above the median, trend is "high"
        return "high" if user_mean > user_median else "low"
