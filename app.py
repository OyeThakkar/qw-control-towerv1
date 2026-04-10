import io
import os
import re
import tempfile

import pandas as pd
from flask import (
    Flask,
    after_this_request,
    render_template,
    request,
    send_file,
    session,
)

app = Flask(__name__)
# SECRET_KEY should be set as a persistent environment variable in production.
# The fallback is suitable only for single-process development use; sessions
# will not survive a server restart without a fixed key.
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# ---------------------------------------------------------------------------
# Theatre Region → Sales Region mapping
# Update this dictionary with the real mapping values as provided by the team.
# ---------------------------------------------------------------------------
THEATRE_REGION_TO_SALES_REGION: dict[str, str] = {
    # North America
    "US - East": "North America",
    "US - West": "North America",
    "US - Central": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    # Europe
    "UK": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Spain": "Europe",
    "Italy": "Europe",
    "Benelux": "Europe",
    "Nordics": "Europe",
    "Eastern Europe": "Europe",
    "Netherlands": "Europe",
    # Asia Pacific
    "India": "Asia Pacific",
    "Australia": "Asia Pacific",
    "New Zealand": "Asia Pacific",
    "Japan": "Asia Pacific",
    "China": "Asia Pacific",
    "South East Asia": "Asia Pacific",
    "Korea": "Asia Pacific",
    # Middle East & Africa
    "Middle East": "Middle East & Africa",
    "Africa": "Middle East & Africa",
    # Latin America
    "Brazil": "Latin America",
    "Latin America": "Latin America",
    # Add additional mappings as needed
}

# ---------------------------------------------------------------------------
# Column name constants (matching the CSV headers exactly)
# ---------------------------------------------------------------------------
COL_THEATRE_NAME = "Theatre Name"
COL_CONTENT_TYPES = "Content Types"
COL_DISTRIBUTOR_COMPANY = "Distributor Company Name"
COL_DISTRIBUTOR_NOTES = "Distributor Delivery Notes"
COL_FEATURE_CPL_NAMES = "Feature: CPL Names"
COL_CONTENT_SIZE = "Content Size in GB"
COL_WORK_ORDER = "Work Order"
COL_THEATRE_REGION = "Theatre Region"
COL_SALES_REGION = "Sales Region"

# Test theatre names (exact match, case-insensitive) – NOT "Happy Cinemas"
TEST_THEATRE_NAMES = {"happy cat", "happy cow", "happy duck"}

# Dummy delivery note keywords (case-insensitive substring match)
DUMMY_NOTE_KEYWORDS = [
    "dummy booking for test",
    "dummy booking for download",
]

# CPL name code → Content Type mapping (checked in priority order)
CPL_CODE_MAP = [
    ({"ADV"}, {"FTR"}, "Feature", "Advert"),   # contains ADV, not FTR, currently Feature
    ({"TLR", "TRL"}, set(), None, "Trailer"),   # contains TLR or TRL (any current type)
    ({"SHR"}, set(), None, "Short"),            # contains SHR
    ({"TSR"}, set(), None, "Teaser"),           # contains TSR
]


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _is_test_theatre(name: str) -> bool:
    """Return True if the theatre name is one of the known test theatres."""
    if pd.isna(name):
        return False
    return name.strip().lower() in TEST_THEATRE_NAMES


def _has_dummy_note(note: str) -> bool:
    """Return True if the delivery note looks like a dummy/test booking."""
    if pd.isna(note):
        return False
    note_lower = note.strip().lower()
    for kw in DUMMY_NOTE_KEYWORDS:
        if kw in note_lower:
            return True
    return False


def _cpl_contains(cpl_name: str, codes: set[str]) -> bool:
    """Return True if any code in *codes* appears as a standalone token in *cpl_name*.

    Tokens are separated by non-alphanumeric characters (underscores, hyphens,
    spaces, etc.).  This prevents partial matches such as "OV" matching inside
    "MOVIE".
    """
    if pd.isna(cpl_name):
        return False
    for code in codes:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])"
        if re.search(pattern, cpl_name, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Main cleanup function
# ---------------------------------------------------------------------------

def apply_cleanup_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Qube Cinema DCP report cleanup rules to *df* in-place.

    The dataframe is modified and returned.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 0. Add Sales Region column right after Theatre Region (if present)
    # ------------------------------------------------------------------
    if COL_THEATRE_REGION in df.columns:
        tr_idx = df.columns.get_loc(COL_THEATRE_REGION)
        # Map theatre region → sales region (unmapped values become empty string)
        sales_region_values = df[COL_THEATRE_REGION].map(
            lambda x: THEATRE_REGION_TO_SALES_REGION.get(str(x).strip(), "")
            if not pd.isna(x)
            else ""
        )
        df.insert(tr_idx + 1, COL_SALES_REGION, sales_region_values)
    else:
        # Theatre Region column is not present; append Sales Region at the end
        df[COL_SALES_REGION] = ""

    # Ensure Content Types column exists (create if missing)
    if COL_CONTENT_TYPES not in df.columns:
        df[COL_CONTENT_TYPES] = ""

    # Work with a string version of Content Types for comparisons
    ct = df[COL_CONTENT_TYPES].fillna("").astype(str)

    # ------------------------------------------------------------------
    # Rule 1: Test theatres → Content Type = TEST
    # ------------------------------------------------------------------
    if COL_THEATRE_NAME in df.columns:
        mask_test_theatre = df[COL_THEATRE_NAME].apply(_is_test_theatre)
        df.loc[mask_test_theatre, COL_CONTENT_TYPES] = "TEST"
        ct = df[COL_CONTENT_TYPES].fillna("").astype(str)

    # ------------------------------------------------------------------
    # Rule 2: Qube Cinema Inc distributor → Content Type = TEST
    # ------------------------------------------------------------------
    if COL_DISTRIBUTOR_COMPANY in df.columns:
        mask_qube = (
            df[COL_DISTRIBUTOR_COMPANY]
            .fillna("")
            .str.strip()
            .str.lower()
            == "qube cinema inc"
        )
        df.loc[mask_qube, COL_CONTENT_TYPES] = "TEST"
        ct = df[COL_CONTENT_TYPES].fillna("").astype(str)

    # ------------------------------------------------------------------
    # Rule 3: Dummy delivery notes → Content Type = TEST
    # ------------------------------------------------------------------
    if COL_DISTRIBUTOR_NOTES in df.columns:
        mask_dummy = df[COL_DISTRIBUTOR_NOTES].apply(_has_dummy_note)
        df.loc[mask_dummy, COL_CONTENT_TYPES] = "TEST"
        ct = df[COL_CONTENT_TYPES].fillna("").astype(str)

    # ------------------------------------------------------------------
    # Rule 4: Propagate dominant content type within each Work Order
    #
    # Priority (highest to lowest):
    #   Feature > Trailer > Advert > Teaser > Short > Promo > (any specific)
    #
    # If any row in an order has "Feature", mark ALL rows in that order
    # as "Feature" (unless already TEST).  Repeat the same logic for
    # the other types in descending priority.
    # ------------------------------------------------------------------
    if COL_WORK_ORDER in df.columns:
        priority_types = ["Feature", "Trailer", "Advert", "Teaser", "Short", "Promo"]
        ct = df[COL_CONTENT_TYPES].fillna("").astype(str)

        for content_type in priority_types:
            # Find all Work Orders that have at least one row with this type
            orders_with_type = df.loc[
                ct.str.strip().str.lower() == content_type.lower(),
                COL_WORK_ORDER,
            ].dropna().unique()

            if len(orders_with_type) == 0:
                continue

            # For those orders, update rows that are NOT already TEST
            mask_order = df[COL_WORK_ORDER].isin(orders_with_type)
            mask_not_test = ct.str.strip().str.lower() != "test"
            df.loc[mask_order & mask_not_test, COL_CONTENT_TYPES] = content_type
            ct = df[COL_CONTENT_TYPES].fillna("").astype(str)

    # ------------------------------------------------------------------
    # Rule 5: Refine based on CPL Name patterns
    # ------------------------------------------------------------------
    if COL_FEATURE_CPL_NAMES in df.columns:
        ct = df[COL_CONTENT_TYPES].fillna("").astype(str)
        cpl = df[COL_FEATURE_CPL_NAMES].fillna("").astype(str)

        for include_codes, exclude_codes, required_ct, new_ct in CPL_CODE_MAP:
            # Build mask: CPL contains any include code
            mask_include = cpl.apply(lambda x: _cpl_contains(x, include_codes))

            # CPL must NOT contain any exclude code
            if exclude_codes:
                mask_exclude = cpl.apply(lambda x: _cpl_contains(x, exclude_codes))
            else:
                mask_exclude = pd.Series(False, index=df.index)

            # Optionally restrict to a specific current content type
            if required_ct is not None:
                mask_ct = ct.str.strip().str.lower() == required_ct.lower()
            else:
                mask_ct = pd.Series(True, index=df.index)

            mask = mask_include & ~mask_exclude & mask_ct
            df.loc[mask, COL_CONTENT_TYPES] = new_ct
            ct = df[COL_CONTENT_TYPES].fillna("").astype(str)

    # ------------------------------------------------------------------
    # Rule 6: Identify VF (Version File) content
    #
    # Conditions (ALL must be true):
    #   a. CPL Name contains "VF" but does NOT contain "OV"
    #   b. Content Size < 30 GB
    #   c. Content Type is blank / NaN
    # ------------------------------------------------------------------
    if COL_FEATURE_CPL_NAMES in df.columns and COL_CONTENT_SIZE in df.columns:
        ct = df[COL_CONTENT_TYPES].fillna("").astype(str)
        cpl = df[COL_FEATURE_CPL_NAMES].fillna("").astype(str)

        mask_vf_cpl = cpl.apply(
            lambda x: _cpl_contains(x, {"VF"}) and not _cpl_contains(x, {"OV"})
        )

        # Coerce content size to numeric; non-numeric → NaN
        size_numeric = pd.to_numeric(df[COL_CONTENT_SIZE], errors="coerce")
        mask_small = size_numeric < 30

        mask_blank_ct = ct.str.strip() == ""

        mask_vf = mask_vf_cpl & mask_small & mask_blank_ct
        df.loc[mask_vf, COL_CONTENT_TYPES] = "VF"

    return df


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a CSV file, apply cleanup rules, and return the cleaned CSV."""
    if "file" not in request.files:
        return render_template("index.html", error="No file part in the request.")

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return render_template("index.html", error="No file selected.")

    filename = uploaded_file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        return render_template(
            "index.html",
            error="Unsupported file format. Please upload a CSV or Excel file.",
        )

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)

        cleaned_df = apply_cleanup_rules(df)

        # Serialize cleaned data to CSV bytes
        output_buffer = io.StringIO()
        cleaned_df.to_csv(output_buffer, index=False)
        csv_bytes = output_buffer.getvalue().encode("utf-8")
        mem_buffer = io.BytesIO(csv_bytes)
        mem_buffer.seek(0)

        # Stats to display after processing
        total_rows = len(cleaned_df)
        test_rows = (
            cleaned_df[COL_CONTENT_TYPES].fillna("").str.strip().str.upper() == "TEST"
        ).sum() if COL_CONTENT_TYPES in cleaned_df.columns else 0

        stats = {
            "total_rows": total_rows,
            "test_rows": int(test_rows),
            "columns": list(cleaned_df.columns),
        }

        # Store the CSV bytes in a temporary file for the download link.
        # NamedTemporaryFile provides OS-managed cleanup and works in
        # containerised / multi-instance environments.
        tmp_fh = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix="dcp_cleaned_"
        )
        try:
            tmp_fh.write(csv_bytes)
        finally:
            tmp_fh.close()
        temp_path = tmp_fh.name

        session["download_file"] = temp_path
        session["download_name"] = (
            "cleaned_" + (uploaded_file.filename or "output.csv")
        )

        return render_template(
            "result.html",
            stats=stats,
            download_ready=True,
        )

    except (ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        return render_template(
            "index.html",
            error=f"Error processing file: {exc}",
        )


@app.route("/download")
def download():
    """Return the cleaned CSV file for download."""
    temp_path = session.get("download_file")
    download_name = session.get("download_name", "cleaned_output.csv")

    if not temp_path or not os.path.exists(temp_path):
        return render_template(
            "index.html",
            error="No cleaned file available. Please upload a file first.",
        )

    @after_this_request
    def _cleanup(response):  # pylint: disable=unused-variable
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return response

    return send_file(
        temp_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=download_name,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
