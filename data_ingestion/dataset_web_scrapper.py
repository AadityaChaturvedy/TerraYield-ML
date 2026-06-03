import pathlib
import sys
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://data.desagri.gov.in/report/crop/horizontal_crop_vertical_year"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
FINAL_OUTPUT = OUTPUT_DIR / "crop_season_year_wide.csv"
TIMEOUT = (10, 180)
VERIFY_TLS = False
SILENCE_INSECURE_WARNING = True

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://data.desagri.gov.in",
    "Referer": "https://data.desagri.gov.in/website/crops-apy-report-web",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-GPC": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Chromium";v="148", "Brave";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

if len(sys.argv) >= 4:
    xsrf_input = sys.argv[1].strip()
    session_input = sys.argv[2].strip()
    TOKEN = sys.argv[3].strip()
    print("Using credentials provided via CLI arguments.")
else:
    print("--- DESAgri Scraper Authentication Setup ---")
    xsrf_input = input("Enter XSRF-TOKEN cookie value: ").strip()
    session_input = input("Enter laravel_session cookie value: ").strip()
    TOKEN = input("Enter the form security token (_token): ").strip()
    print("--------------------------------------------\n")

if not xsrf_input or not session_input or not TOKEN:
    print("\n[Error] Missing required credentials.")
    print("Usage via CLI: python script.py <XSRF-TOKEN> <laravel_session> <_token>")
    print("Or run without arguments to enter them interactively.")
    sys.exit(1)

COOKIES = {
    'XSRF-TOKEN': xsrf_input,
    'laravel_session': session_input
}

SEASON_CODES = ["R", "K", "A", "W", "S", "Y"]
SEASON_LABELS = {
    "R": "Rabi",
    "K": "Kharif",
    "A": "Autum",
    "W": "Winter",
    "S": "Summer",
    "Y": "Year",
}
SEASON_NAME_MAP = {
    "rabi": "Rabi",
    "kharif": "Kharif",
    "autumn": "Autum",
    "summer": "Summer",
    "winter": "Winter",
    "year": "Year",
    "whole year": "Year",
}
CROPS = {
    "1": "rice",
    "5": "maize",
    "2": "wheat",
    "14": "arhar_tur",
    "15": "groundnut",
    "24": "soyabean",
    "26": "cotton_lint",
    "22": "coconut",
    "62": "sugarcane",
    "122": "ginger",
    "63": "tobacco",
    "124": "banana",
    "49": "onion",
    "45": "potato",
    "41": "turmeric",
}

START_YEAR = 1997
END_YEAR = 2022


def build_payload(crop_code: str, season_code: str, year: int) -> dict[str, str]:
    return {
        "reportformat": "horizontal_crop_vertical_year",
        "fltrstates[]": "",
        "fltrdistricts[]": "all",
        "fltrcrops[]": crop_code,
        "fltrseason[]": season_code,
        "fltrstartyear": str(year),
        "fltrendyear": str(year),
        "fltrrptformat": "exl",
        "_token": TOKEN,
    }


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_existing_crop_outputs() -> set[str]:
    existing = set()
    for path in OUTPUT_DIR.glob("crop_*_season_year_wide.csv"):
        name = path.name
        prefix = "crop_"
        suffix = "_season_year_wide.csv"
        if name.startswith(prefix) and name.endswith(suffix):
            crop_name = name[len(prefix) : -len(suffix)]
            if crop_name:
                existing.add(crop_name)
    return existing


def save_response_to_file(response: requests.Response, path: pathlib.Path) -> bytes:
    first_chunk = b""
    with path.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            if not first_chunk:
                first_chunk = chunk
            file_handle.write(chunk)
    return first_chunk


def is_html_response(response: requests.Response, first_chunk: bytes) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        return True
    sniff = first_chunk.lstrip()[:20].lower()
    return sniff.startswith(b"<") or b"<!doctype html" in sniff


def read_html_table(html_path: pathlib.Path) -> pd.DataFrame:
    try:
        tables = pd.read_html(html_path, attrs={"id": "apyreport"})
        if tables:
            return tables[0]
    except ValueError:
        pass

    try:
        tables = pd.read_html(html_path)
    except ValueError as exc:
        raise ValueError(
            "No HTML tables found. Install lxml or html5lib if needed."
        ) from exc

    if not tables:
        raise ValueError("No HTML tables found in response.")
    return tables[0]


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    return frame


def normalize_season_token(token: str) -> str:
    token = token.strip().lower()
    if token == "whole year":
        return "year"
    return token


def normalize_location_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^\s*\d+\.\s+", "", regex=True).str.strip()


def normalize_year_series(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        flattened = []
        for levels in frame.columns:
            parts = [str(level).strip() for level in levels]
            parts = [part for part in parts if part and not part.lower().startswith("unnamed")]
            if not parts:
                flattened.append("")
                continue
            lowered = [part.lower() for part in parts]
            if len(set(lowered)) == 1:
                flattened.append(parts[0])
                continue
            flattened.append("|".join(parts))
        frame.columns = flattened
    return frame


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        for column in frame.columns:
            if column == candidate:
                return column
            if column.startswith(f"{candidate}|"):
                return column
            if column.endswith(f"|{candidate}"):
                return column
            if f"|{candidate}|" in column:
                return column
    raise ValueError(f"Missing expected column: {candidates}")


def rebuild_header_from_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[0] < 2:
        return frame

    row0 = frame.iloc[0].astype(str).str.strip().str.lower()
    row1 = frame.iloc[1].astype(str).str.strip().str.lower()
    if (row0 == "no data found").all():
        return frame.iloc[0:0]
    season_tokens = {"rabi", "kharif", "autumn", "summer", "winter", "year", "whole year"}
    has_season = any(token in season_tokens for token in row0.values)
    has_metric = any(
        "area" in str(value) or "production" in str(value) or "yield" in str(value)
        for value in row1.values
    )

    if not (has_season and has_metric):
        return frame

    new_columns = []
    for idx, col in enumerate(frame.columns):
        if idx < 3:
            name = row0.iloc[idx]
            new_columns.append(name if name else str(col).strip().lower())
            continue

        season = normalize_season_token(row0.iloc[idx])
        metric = row1.iloc[idx]
        if season and metric:
            new_columns.append(f"{season}|{metric}")
        else:
            new_columns.append(str(col).strip().lower())

    rebuilt = frame.iloc[2:].copy()
    rebuilt.columns = new_columns
    return rebuilt


def read_excel_with_best_header(xls_path: pathlib.Path) -> pd.DataFrame:
    frame = pd.read_excel(xls_path, engine="xlrd", header=[0, 1])
    if isinstance(frame.columns, pd.MultiIndex):
        level_1 = [str(col).strip().lower() for col in frame.columns.get_level_values(1)]
        has_named_subheaders = any(not col.startswith("unnamed") for col in level_1)
        if has_named_subheaders:
            return frame
    return pd.read_excel(xls_path, engine="xlrd", header=0)


def convert_xls_to_csv(xls_path: pathlib.Path, csv_path: pathlib.Path) -> pd.DataFrame:
    frame = read_excel_with_best_header(xls_path)
    frame.to_csv(csv_path, index=False)
    return frame


def extract_wide_season_rows(
    frame: pd.DataFrame, crop_name: str, state_col: str, district_col: str, year_col: str
) -> pd.DataFrame:
    records = []
    for season_key, season_label in SEASON_NAME_MAP.items():
        area_col = None
        production_col = None
        yield_col = None
        for col in frame.columns:
            parts = [part.strip().lower() for part in str(col).split("|") if part.strip()]
            if season_key not in parts:
                continue
            metric = next((part for part in parts if "area" in part), "")
            if metric:
                area_col = col
            metric = next((part for part in parts if "production" in part), "")
            if metric:
                production_col = col
            metric = next((part for part in parts if "yield" in part), "")
            if metric:
                yield_col = col

        if not any([area_col, production_col, yield_col]):
            continue

        records.append(
            pd.DataFrame(
                {
                    "Crop_Name": crop_name,
                    "State": normalize_location_series(frame[state_col]),
                    "District": normalize_location_series(frame[district_col]),
                    "Year": normalize_year_series(frame[year_col]),
                    "Season": season_label,
                    "Area": frame[area_col] if area_col else pd.NA,
                    "Production": frame[production_col] if production_col else pd.NA,
                    "Yield": frame[yield_col] if yield_col else pd.NA,
                }
            )
        )

    if not records:
        raise ValueError("No season metric columns found in wide header format.")

    return pd.concat(records, ignore_index=True)


def standardize_frame(
    frame: pd.DataFrame, crop_name: str, season_code: str
) -> pd.DataFrame:
    frame = rebuild_header_from_rows(frame)
    frame = normalize_columns(flatten_columns(frame))

    if frame.empty:
        return frame
    if (frame.iloc[0].astype(str).str.strip().str.lower() == "no data found").all():
        return frame.iloc[0:0]

    state_col = find_column(frame, ["state", "state_name"])
    district_col = find_column(frame, ["district", "district_name"])
    year_col = find_column(frame, ["year"])

    if any("|" in col for col in frame.columns):
        return extract_wide_season_rows(frame, crop_name, state_col, district_col, year_col)

    area_col = find_column(frame, ["area"])
    production_col = find_column(frame, ["production"])
    yield_col = find_column(frame, ["yield"])

    return pd.DataFrame(
        {
            "Crop_Name": crop_name,
            "State": normalize_location_series(frame[state_col]),
            "District": normalize_location_series(frame[district_col]),
            "Year": normalize_year_series(frame[year_col]),
            "Season": SEASON_LABELS[season_code],
            "Area": frame[area_col],
            "Production": frame[production_col],
            "Yield": frame[yield_col],
        }
    )


def fetch_report(
    session: requests.Session,
    crop_code: str,
    crop_name: str,
    season_code: str,
    year: int,
) -> pd.DataFrame:
    payload = build_payload(crop_code, season_code, year)
    response = session.post(
        BASE_URL,
        cookies=COOKIES,
        headers=HEADERS,
        data=payload,
        verify=VERIFY_TLS,
        stream=True,
        timeout=TIMEOUT,
    )
    if response.status_code == 419:
        print(
            "Received 419 from server (likely expired token/cookies). "
            f"Skipping {crop_name} {season_code} {year}."
        )
        return pd.DataFrame()

    response.raise_for_status()

    xls_path = OUTPUT_DIR / f"crop_{crop_name}_{season_code}_{year}.xls"
    csv_path = OUTPUT_DIR / f"crop_{crop_name}_{season_code}_{year}.csv"
    first_chunk = save_response_to_file(response, xls_path)
    if csv_path.exists():
        csv_path.unlink()
    if is_html_response(response, first_chunk):
        html_path = xls_path.with_suffix(".html")
        xls_path.replace(html_path)
        raw_frame = read_html_table(html_path)
        html_path.unlink(missing_ok=True)
        print(f"Parsed HTML response for {crop_name} {season_code} {year}")
        return standardize_frame(raw_frame, crop_name, season_code)
    raw_frame = convert_xls_to_csv(xls_path, csv_path)
    xls_path.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)
    print(f"Parsed XLS response for {crop_name} {season_code} {year}")
    return standardize_frame(raw_frame, crop_name, season_code)


def pivot_season_data(all_frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(all_frames, ignore_index=True)
    pivoted = combined.pivot_table(
        index=["Crop_Name", "State", "District", "Year"],
        columns="Season",
        values=["Area", "Production", "Yield"],
        aggfunc="first",
    )

    # Build requested column order: Season_Metric
    ordered_columns = []
    for season in ["Rabi", "Kharif", "Autum", "Winter", "Summer", "Year"]:
        for metric in ["Area", "Production", "Yield"]:
            ordered_columns.append((metric, season))

    pivoted = pivoted.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    pivoted.columns = [f"{season}_{metric}" for metric, season in pivoted.columns]
    pivoted = pivoted.reset_index()
    return pivoted


def main() -> None:
    if SILENCE_INSECURE_WARNING and not VERIFY_TLS:
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    ensure_output_dir()
    existing_outputs = get_existing_crop_outputs()

    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    for crop_code, crop_name in CROPS.items():
        if crop_name in existing_outputs:
            print(f"Skipping {crop_name}; output already exists.")
            continue
        crop_frames: list[pd.DataFrame] = []
        for season_code in SEASON_CODES:
            for year in range(START_YEAR, END_YEAR + 1):
                standardized = fetch_report(session, crop_code, crop_name, season_code, year)
                if not standardized.empty:
                    crop_frames.append(standardized)

        if not crop_frames:
            continue
        final_frame = pivot_season_data(crop_frames).fillna(-1)
        crop_output = OUTPUT_DIR / f"crop_{crop_name}_season_year_wide.csv"
        final_frame.to_csv(crop_output, index=False)
        print(f"Saved crop CSV to {crop_output}")


if __name__ == "__main__":
    main()