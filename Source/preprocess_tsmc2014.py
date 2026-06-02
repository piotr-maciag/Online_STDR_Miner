from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_COLUMNS = [
    "User_ID",
    "Venue_ID",
    "Venue_category_ID",
    "Event_type",
    "latitude",
    "longitude",
    "Timezone_offset_minutes",
    "UTC_time",
]

OUTPUT_COLUMNS = [
    "Event_ID",
    "longitude",
    "latitude",
    "timestamp",
    "Occurrence_time",
    "Event_type",
]
TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def preprocess_tsmc2014_file(
    raw_path: Path | str,
    output_path: Path | str,
    encoding: str = "latin-1",
) -> Path:
    """Convert one TSMC2014 TSV file to the project event CSV format."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    data = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        encoding=encoding,
    )
    parsed_time = pd.to_datetime(data["UTC_time"], format=TIME_FORMAT, utc=True)

    output = pd.DataFrame(
        {
            "Event_ID": range(1, len(data) + 1),
            "longitude": data["longitude"].astype(float),
            "latitude": data["latitude"].astype(float),
            "timestamp": (parsed_time - parsed_time.min()).dt.total_seconds() / 60.0,
            "Occurrence_time": data["UTC_time"],
            "Event_type": data["Event_type"],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, columns=OUTPUT_COLUMNS)
    return output_path


def preprocess_all(
    raw_dir: Path | str = Path("Data") / "Raw",
    output_dir: Path | str = Path("Data") / "Preprocessed",
) -> list[Path]:
    """Preprocess every raw TSMC2014 city file in raw_dir."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    raw_files = sorted(raw_dir.glob("dataset_TSMC2014_*.txt"))
    raw_files = [path for path in raw_files if "readme" not in path.name.lower()]

    output_paths = []
    for raw_file in raw_files:
        city = raw_file.stem.removeprefix("dataset_TSMC2014_")
        output_path = output_dir / f"dataset_TSMC2014_{city}_preprocessed.csv"
        output_paths.append(preprocess_tsmc2014_file(raw_file, output_path))

    return output_paths


if __name__ == "__main__":
    for path in preprocess_all():
        print(path)
