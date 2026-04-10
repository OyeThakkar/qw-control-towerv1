"""Unit tests for the DCP report cleanup rules in app.py."""
import pandas as pd
import pytest

from app import (
    COL_CONTENT_TYPES,
    COL_CONTENT_SIZE,
    COL_DISTRIBUTOR_COMPANY,
    COL_DISTRIBUTOR_NOTES,
    COL_FEATURE_CPL_NAMES,
    COL_SALES_REGION,
    COL_THEATRE_NAME,
    COL_THEATRE_REGION,
    COL_WORK_ORDER,
    apply_cleanup_rules,
)


def make_df(**kwargs) -> pd.DataFrame:
    """Helper: create a 1-row dataframe with defaults for all required columns."""
    defaults = {
        COL_THEATRE_NAME: "Normal Theatre",
        COL_CONTENT_TYPES: "",
        COL_DISTRIBUTOR_COMPANY: "Some Distributor",
        COL_DISTRIBUTOR_NOTES: "",
        COL_FEATURE_CPL_NAMES: "",
        COL_CONTENT_SIZE: "50",
        COL_WORK_ORDER: "WO-001",
        COL_THEATRE_REGION: "",
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Rule 1 – Test theatre names
# ---------------------------------------------------------------------------

class TestRule1TestTheatres:
    def test_happy_cat_marked_test(self):
        df = make_df(**{COL_THEATRE_NAME: "Happy Cat"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_happy_cow_marked_test(self):
        df = make_df(**{COL_THEATRE_NAME: "Happy Cow"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_happy_duck_marked_test(self):
        df = make_df(**{COL_THEATRE_NAME: "Happy Duck"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_happy_cinemas_not_marked_test(self):
        df = make_df(**{COL_THEATRE_NAME: "Happy Cinemas"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] != "TEST"

    def test_case_insensitive_match(self):
        df = make_df(**{COL_THEATRE_NAME: "HAPPY CAT"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_normal_theatre_unchanged(self):
        df = make_df(**{COL_THEATRE_NAME: "Grand Cinema", COL_CONTENT_TYPES: "Feature"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Feature"


# ---------------------------------------------------------------------------
# Rule 2 – Qube Cinema Inc
# ---------------------------------------------------------------------------

class TestRule2QubeCinemaInc:
    def test_qube_cinema_inc_marked_test(self):
        df = make_df(**{COL_DISTRIBUTOR_COMPANY: "Qube Cinema Inc"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_case_insensitive(self):
        df = make_df(**{COL_DISTRIBUTOR_COMPANY: "qube cinema inc"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_other_distributor_unchanged(self):
        df = make_df(**{COL_DISTRIBUTOR_COMPANY: "Warner Bros"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] != "TEST"


# ---------------------------------------------------------------------------
# Rule 3 – Dummy delivery notes
# ---------------------------------------------------------------------------

class TestRule3DummyNotes:
    def test_dummy_booking_for_test(self):
        df = make_df(**{COL_DISTRIBUTOR_NOTES: "Dummy booking for Test"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_dummy_booking_for_download(self):
        df = make_df(**{COL_DISTRIBUTOR_NOTES: "Dummy booking for Download"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_case_insensitive_note(self):
        df = make_df(**{COL_DISTRIBUTOR_NOTES: "DUMMY BOOKING FOR TEST - please ignore"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "TEST"

    def test_real_note_unchanged(self):
        df = make_df(**{COL_DISTRIBUTOR_NOTES: "Please deliver by Friday 5pm"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] != "TEST"


# ---------------------------------------------------------------------------
# Rule 4 – Content type propagation within Work Order
# ---------------------------------------------------------------------------

class TestRule4ContentTypePropagation:
    def test_feature_propagates_to_blank_rows(self):
        df = pd.DataFrame([
            {COL_WORK_ORDER: "WO-100", COL_CONTENT_TYPES: "Feature",
             COL_THEATRE_NAME: "Cinema A", COL_DISTRIBUTOR_COMPANY: "Dist",
             COL_DISTRIBUTOR_NOTES: "", COL_FEATURE_CPL_NAMES: "", COL_CONTENT_SIZE: "50",
             COL_THEATRE_REGION: ""},
            {COL_WORK_ORDER: "WO-100", COL_CONTENT_TYPES: "",
             COL_THEATRE_NAME: "Cinema A", COL_DISTRIBUTOR_COMPANY: "Dist",
             COL_DISTRIBUTOR_NOTES: "", COL_FEATURE_CPL_NAMES: "", COL_CONTENT_SIZE: "50",
             COL_THEATRE_REGION: ""},
        ])
        result = apply_cleanup_rules(df)
        assert result.iloc[1][COL_CONTENT_TYPES] == "Feature"

    def test_test_rows_not_overwritten_by_propagation(self):
        df = pd.DataFrame([
            {COL_WORK_ORDER: "WO-200", COL_CONTENT_TYPES: "Feature",
             COL_THEATRE_NAME: "Cinema B", COL_DISTRIBUTOR_COMPANY: "Dist",
             COL_DISTRIBUTOR_NOTES: "", COL_FEATURE_CPL_NAMES: "", COL_CONTENT_SIZE: "50",
             COL_THEATRE_REGION: ""},
            {COL_WORK_ORDER: "WO-200", COL_CONTENT_TYPES: "TEST",
             COL_THEATRE_NAME: "Happy Cat", COL_DISTRIBUTOR_COMPANY: "Dist",
             COL_DISTRIBUTOR_NOTES: "", COL_FEATURE_CPL_NAMES: "", COL_CONTENT_SIZE: "50",
             COL_THEATRE_REGION: ""},
        ])
        result = apply_cleanup_rules(df)
        assert result.iloc[1][COL_CONTENT_TYPES] == "TEST"


# ---------------------------------------------------------------------------
# Rule 5 – CPL Name refinement
# ---------------------------------------------------------------------------

class TestRule5CPLRefinement:
    def test_adv_not_ftr_changes_feature_to_advert(self):
        df = make_df(**{
            COL_FEATURE_CPL_NAMES: "MOVIE_ADV_EN_3D",
            COL_CONTENT_TYPES: "Feature",
        })
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Advert"

    def test_adv_with_ftr_stays_feature(self):
        df = make_df(**{
            COL_FEATURE_CPL_NAMES: "MOVIE_FTR_ADV_EN",
            COL_CONTENT_TYPES: "Feature",
        })
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Feature"

    def test_tlr_changes_to_trailer(self):
        df = make_df(**{COL_FEATURE_CPL_NAMES: "MOVIE_TLR_EN"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Trailer"

    def test_trl_changes_to_trailer(self):
        df = make_df(**{COL_FEATURE_CPL_NAMES: "MOVIE_TRL_EN"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Trailer"

    def test_shr_changes_to_short(self):
        df = make_df(**{COL_FEATURE_CPL_NAMES: "CONTENT_SHR_2024"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Short"

    def test_tsr_changes_to_teaser(self):
        df = make_df(**{COL_FEATURE_CPL_NAMES: "FILM_TSR_EN_4K"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Teaser"


# ---------------------------------------------------------------------------
# Rule 6 – VF content
# ---------------------------------------------------------------------------

class TestRule6VFContent:
    def test_vf_small_blank_type_marked_vf(self):
        df = make_df(**{
            COL_FEATURE_CPL_NAMES: "MOVIE_VF_FR",
            COL_CONTENT_SIZE: "25",
            COL_CONTENT_TYPES: "",
        })
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "VF"

    def test_vf_with_ov_not_marked_vf(self):
        df = make_df(**{
            COL_FEATURE_CPL_NAMES: "MOVIE_VF_OV_EN",
            COL_CONTENT_SIZE: "25",
            COL_CONTENT_TYPES: "",
        })
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] != "VF"

    def test_vf_large_size_not_marked_vf(self):
        df = make_df(**{
            COL_FEATURE_CPL_NAMES: "MOVIE_VF_FR",
            COL_CONTENT_SIZE: "45",
            COL_CONTENT_TYPES: "",
        })
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] != "VF"

    def test_vf_with_existing_type_not_overwritten(self):
        df = make_df(**{
            COL_FEATURE_CPL_NAMES: "MOVIE_VF_FR",
            COL_CONTENT_SIZE: "20",
            COL_CONTENT_TYPES: "Feature",
        })
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_CONTENT_TYPES] == "Feature"


# ---------------------------------------------------------------------------
# Sales Region column insertion
# ---------------------------------------------------------------------------

class TestSalesRegionColumn:
    def test_sales_region_inserted_after_theatre_region(self):
        df = make_df(**{COL_THEATRE_REGION: "India"})
        result = apply_cleanup_rules(df)
        cols = list(result.columns)
        tr_idx = cols.index(COL_THEATRE_REGION)
        assert cols[tr_idx + 1] == COL_SALES_REGION

    def test_sales_region_mapped_correctly(self):
        df = make_df(**{COL_THEATRE_REGION: "India"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_SALES_REGION] == "Asia Pacific"

    def test_unknown_region_blank_sales_region(self):
        df = make_df(**{COL_THEATRE_REGION: "UnknownRegion"})
        result = apply_cleanup_rules(df)
        assert result.iloc[0][COL_SALES_REGION] == ""

    def test_sales_region_appended_when_no_theatre_region(self):
        df = pd.DataFrame([{
            COL_THEATRE_NAME: "Cinema",
            COL_CONTENT_TYPES: "",
            COL_DISTRIBUTOR_COMPANY: "Dist",
            COL_DISTRIBUTOR_NOTES: "",
            COL_FEATURE_CPL_NAMES: "",
            COL_CONTENT_SIZE: "50",
            COL_WORK_ORDER: "WO-999",
        }])
        result = apply_cleanup_rules(df)
        assert COL_SALES_REGION in result.columns
