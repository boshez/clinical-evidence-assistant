"""Tests for the temporal rule in patient_drug_safety.

Run with:  python -m graph.test_safety
They use the hand-made TEST_ patients created by graph/load_patients.py.

Every "should not flag" test also checks that the SAME patient DOES flag on a
date when the condition was active. Without that, a test could pass simply
because an edge is missing from the graph, not because the date rule works.
"""
from datetime import date

from neo4j import Driver

from graph.connect import get_driver
from graph.safety import patient_drug_safety

AS_OF = date(2026, 6, 1)
# Spironolactone is contraindicated in hyperkalemia (approved edge from its label).
DRUG = "spironolactone"


def flagged(driver: Driver, patient_id: str, drug: str, as_of: date) -> bool:
    """Shortcut: was anything flagged for this patient, drug and date?"""
    return patient_drug_safety(driver, patient_id, drug, as_of)["flagged"]


def test_active_condition_flags(driver: Driver) -> None:
    """1. Condition started before as_of and has not ended: flag."""
    assert flagged(driver, "TEST_ACTIVE", DRUG, AS_OF)


def test_resolved_condition_does_not_flag(driver: Driver) -> None:
    """2. Condition ended in Feb 2025: no flag in June 2026."""
    assert not flagged(driver, "TEST_RESOLVED", DRUG, AS_OF)
    # Same patient, a date inside the episode: it must flag.
    assert flagged(driver, "TEST_RESOLVED", DRUG, date(2025, 1, 20))


def test_future_condition_does_not_flag(driver: Driver) -> None:
    """3. Condition starts in July 2026: no flag in June 2026."""
    assert not flagged(driver, "TEST_FUTURE", DRUG, AS_OF)
    assert flagged(driver, "TEST_FUTURE", DRUG, date(2026, 8, 1))


def test_resolved_on_as_of_does_not_flag(driver: Driver) -> None:
    """4. Boundary: resolved exactly on as_of counts as ended."""
    assert not flagged(driver, "TEST_BOUNDARY", DRUG, AS_OF)
    # The day before, it was still active.
    assert flagged(driver, "TEST_BOUNDARY", DRUG, date(2026, 5, 31))
    # Onset day itself counts as active (onset <= as_of).
    assert flagged(driver, "TEST_BOUNDARY", DRUG, date(2026, 5, 1))


def test_current_medication_flags_interaction(driver: Driver) -> None:
    """5. Patient takes warfarin now: ibuprofen is flagged as an interaction."""
    result = patient_drug_safety(driver, "TEST_TAKES_NOW", "ibuprofen", AS_OF)
    assert [i["drug"] for i in result["interactions"]] == ["warfarin"]


def test_stopped_medication_does_not_flag(driver: Driver) -> None:
    """6. Warfarin was stopped in March 2025: no interaction flag in 2026."""
    assert not flagged(driver, "TEST_TAKES_STOPPED", "ibuprofen", AS_OF)
    assert flagged(driver, "TEST_TAKES_STOPPED", "ibuprofen", date(2025, 2, 1))


def test_unknown_patient_is_an_error(driver: Driver) -> None:
    """7. An unknown patient must raise, never return 'no flags'."""
    try:
        patient_drug_safety(driver, "NO_SUCH_PATIENT", DRUG, AS_OF)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown patient")


TESTS = [
    test_active_condition_flags,
    test_resolved_condition_does_not_flag,
    test_future_condition_does_not_flag,
    test_resolved_on_as_of_does_not_flag,
    test_current_medication_flags_interaction,
    test_stopped_medication_does_not_flag,
    test_unknown_patient_is_an_error,
]

if __name__ == "__main__":
    failures = 0
    with get_driver() as driver:
        for test in TESTS:
            try:
                test(driver)
                print("PASS", test.__name__)
            except AssertionError as error:
                failures += 1
                print("FAIL", test.__name__, error)
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(1 if failures else 0)
