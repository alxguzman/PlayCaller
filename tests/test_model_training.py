import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.data.features import add_team_features
from src.model.evaluate import model_recommendations
from src.model.train import augment_optional_context, mask_optional_context


class TrainingTests(unittest.TestCase):
    def test_missing_history_has_no_fabricated_api_rank(self):
        from src.api.main import _rank_table
        table = pd.DataFrame({"off_epa_per_play": [np.nan, 0.2]}, index=["KC", "BUF"])
        result = _rank_table(table, [("overall", "off_epa_per_play", False)])
        self.assertNotIn("KC", result["ranks"])
        self.assertEqual(result["ranks"]["BUF"]["overall_rank"], 1)

    def test_optional_context_mask_preserves_situation_and_original_input(self):
        X = pd.DataFrame({"down": [1, 3], "defenders_in_box": [7., 8.],
                          "box_x_run": [7., 0.], "box_x_pass": [0., 8.],
                          "personnel_11": [True, False], "play_concept_short_pass": [False, True]})
        original = X.copy()
        masked = mask_optional_context(X)
        self.assertTrue(masked[["defenders_in_box", "box_x_run", "box_x_pass"]].isna().all().all())
        self.assertFalse(masked.personnel_11.any())
        pd.testing.assert_frame_equal(X, original)
        pd.testing.assert_series_equal(masked.down, X.down)
        pd.testing.assert_series_equal(masked.play_concept_short_pass, X.play_concept_short_pass)
        augmented, labels, weights = augment_optional_context(X, pd.Series([0, 1]))
        self.assertEqual(len(augmented), 4)
        self.assertEqual(list(labels), [0, 1, 0, 1])
        np.testing.assert_allclose(weights[:2] + weights[2:], np.ones(2))

    def test_future_games_do_not_fill_early_week_history(self):
        df = pd.DataFrame({
            "season": [2025] * 3, "week": [1, 2, 3], "posteam": ["KC"] * 3,
            "defteam": ["BUF"] * 3, "epa": [1., 3., 100.], "success": [1, 1, 1],
            "yards_gained": [10., 20., 30.], "play_type": ["pass"] * 3,
            "down": [1] * 3, "yardline_100": [50] * 3, "qb_dropback": [1] * 3,
            "passer_player_id": ["QB1"] * 3, "qb_epa": [1., 3., 100.],
            "cpoe": [1., 2., 3.], "sack": [0] * 3,
        })
        before = add_team_features(df, save=False)
        df.loc[2, ["epa", "qb_epa"]] = 999.
        after = add_team_features(df, save=False)
        columns = [c for c in before.columns if c.startswith(("off_", "def_", "qbf_"))]
        pd.testing.assert_frame_equal(before[columns], after[columns])
        self.assertTrue(np.isnan(before.loc[0, "off_epa_per_play"]))
        self.assertEqual(before.loc[1, "off_epa_per_play"], 1.)
        self.assertEqual(before.loc[2, "off_epa_per_play"], 2.)

    def test_counterfactual_calls_recompute_box_interactions(self):
        X = pd.DataFrame({"play_concept_inside_run": [True], "play_concept_short_pass": [False],
                          "down": [1], "ydstogo": [10], "defenders_in_box": [8.],
                          "box_x_run": [8.], "box_x_pass": [0.]})
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.5, 0.5]])
        epa = MagicMock()
        def predict(rows):
            run = bool(rows.iloc[0].play_concept_inside_run)
            self.assertEqual(rows.iloc[0].box_x_run, 8. if run else 0.)
            self.assertEqual(rows.iloc[0].box_x_pass, 0. if run else 8.)
            return np.array([0.1])
        epa.predict.side_effect = predict
        model_recommendations(X, {"model": model, "epa_model": epa,
                                 "concept_classes": ["inside_run", "short_pass"]})
        self.assertEqual(epa.predict.call_count, 2)


if __name__ == "__main__":
    unittest.main()
