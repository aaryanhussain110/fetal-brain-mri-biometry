from __future__ import annotations

from math import erf, sqrt
import unittest

import main
from data_reference import LOOKUP_TABLES, PARAMETERS, lookup_distribution


class CalculatorTests(unittest.TestCase):
    @staticmethod
    def normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + erf(value / sqrt(2.0)))

    def parameter(self, parameter_id: str):
        return next(
            parameter
            for parameter in PARAMETERS
            if parameter.parameter_id == parameter_id
        )

    def test_tdpf_distribution_matches_document_formula_at_24_weeks(self) -> None:
        mean, std_dev = main.analytic_distribution("tdpf", 24.0)
        self.assertAlmostEqual(mean, 32.09872, places=5)
        self.assertAlmostEqual(std_dev, 2.15884, places=5)

    def test_csa_distribution_matches_document_formula_at_24_weeks(self) -> None:
        mean, std_dev = main.analytic_distribution("csa", 24.0)
        self.assertAlmostEqual(mean, 75.16904, places=5)
        self.assertAlmostEqual(std_dev, 6.25636, places=5)

    def test_worked_example_z_scores_match_formula_derived_values(self) -> None:
        tdpf_result = main.evaluate_parameter(self.parameter("tdpf"), "24.0", 24.0)
        csa_result = main.evaluate_parameter(self.parameter("csa"), "55.0", 24.0)

        self.assertIsNotNone(tdpf_result.z_score)
        self.assertIsNotNone(csa_result.z_score)
        self.assertAlmostEqual(tdpf_result.z_score, -3.7514220599951824, places=6)
        self.assertAlmostEqual(csa_result.z_score, -3.223765895824411, places=6)
        self.assertAlmostEqual(
            tdpf_result.percentile,
            self.normal_cdf(tdpf_result.z_score) * 100.0,
            places=10,
        )
        self.assertAlmostEqual(
            csa_result.percentile,
            self.normal_cdf(csa_result.z_score) * 100.0,
            places=10,
        )

    def test_mahalanobis_classifier_strongly_favors_ontd_for_worked_example(self) -> None:
        tdpf_result = main.evaluate_parameter(self.parameter("tdpf"), "24.0", 24.0)
        csa_result = main.evaluate_parameter(self.parameter("csa"), "55.0", 24.0)

        posterior = main.chiari_joint_probability(tdpf_result, csa_result)
        self.assertIsNotNone(posterior)
        self.assertGreater(posterior, 0.99)

    def test_flagged_sample_triggers_both_document_based_warning_cards(self) -> None:
        form_values = main.default_form_values()
        form_values.update(
            {
                "ga_weeks": "24",
                "ga_days": "0",
                "left_ventricular_diameter": "16.0",
                "right_ventricular_diameter": "15.2",
                "tdpf": "24.0",
                "csa": "55.0",
            }
        )

        context = main.build_page_context(form_values)
        warning_titles = {card["title"] for card in context["warning_cards"]}

        self.assertIn("Severe Ventriculomegaly", warning_titles)
        self.assertIn("Chiari II Malformation / Open NTD", warning_titles)

    def test_normal_sample_has_no_warning_cards(self) -> None:
        form_values = main.default_form_values()
        form_values.update(
            {
                "ga_weeks": "21",
                "ga_days": "0",
                "brain_bpd": "48.6",
                "tdpf": "26.1960",
                "csa": "68.9924",
            }
        )

        context = main.build_page_context(form_values)
        self.assertEqual(context["warning_cards"], [])

    def test_severe_ventriculomegaly_threshold_is_inclusive_at_15_mm(self) -> None:
        form_values = main.default_form_values()
        form_values.update(
            {
                "ga_weeks": "24",
                "ga_days": "0",
                "left_ventricular_diameter": "15.0",
            }
        )

        context = main.build_page_context(form_values)
        warning_titles = {card["title"] for card in context["warning_cards"]}
        self.assertIn("Severe Ventriculomegaly", warning_titles)

    def test_documented_brain_bpd_rows_are_preserved(self) -> None:
        rows_by_week = {
            int(row["ga_weeks"]): row
            for row in LOOKUP_TABLES["brain_bpd"]
        }

        self.assertEqual(rows_by_week[20]["centile_5"], 41.2)
        self.assertEqual(rows_by_week[20]["centile_50"], 45.1)
        self.assertEqual(rows_by_week[20]["centile_95"], 49.0)
        self.assertEqual(rows_by_week[21]["centile_5"], 44.5)
        self.assertEqual(rows_by_week[21]["centile_50"], 48.6)
        self.assertEqual(rows_by_week[21]["centile_95"], 52.7)

    def test_lookup_distributions_are_safe_for_all_parameters(self) -> None:
        for parameter_id in LOOKUP_TABLES:
            distribution = lookup_distribution(parameter_id, 24.0)
            self.assertIsNotNone(distribution, msg=f"{parameter_id} returned no distribution")
            mean, std_dev = distribution
            self.assertGreater(mean, 0.0, msg=f"{parameter_id} returned non-positive mean")
            self.assertGreater(std_dev, 0.0, msg=f"{parameter_id} returned non-positive std dev")


if __name__ == "__main__":
    unittest.main()
