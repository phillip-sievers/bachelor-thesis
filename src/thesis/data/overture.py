"""Cached Overture extraction with provenance for source and union labels (D03, D07, D17)."""

import os
from pathlib import Path

import duckdb as db

from thesis.config import REPO_ROOT


def connect(cache: Path) -> db.DuckDBPyConnection:
    con = db.connect(str(cache))
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    return con


def bbox_filter(boxes: list[dict]) -> str:
    """OR containment tests, sfast pruning"""
    return " OR ".join(
        "(bbox.xmin >= {xmin} AND bbox.xmax <= {xmax}  AND bbox.ymin >= {ymin} AND bbox.ymax <= {ymax})".format(
            xmin=float(bbox["xmin"]),
            ymin=float(bbox["ymin"]),
            xmax=float(bbox["xmax"]),
            ymax=float(bbox["ymax"]),
        )
        for bbox in boxes
    )


def replace_boxes(con, boxes: list[dict]) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE boxes (
            name VARCHAR,
            grid_x INTEGER,
            grid_y INTEGER,
            ecoregion_code VARCHAR,
            ecoregion_name VARCHAR,
            state VARCHAR,
            candidate_cells_in_ecoregion INTEGER,
            selected_cells_in_ecoregion INTEGER,
            cell_selection_probability DOUBLE,
            xmin DOUBLE,
            ymin DOUBLE,
            xmax DOUBLE,
            ymax DOUBLE
        )
    """)
    con.executemany(
        "INSERT INTO boxes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                bbox["name"],
                bbox["grid_x"],
                bbox["grid_y"],
                bbox["ecoregion_code"],
                bbox["ecoregion_name"],
                bbox["state"],
                bbox["candidate_cells_in_ecoregion"],
                bbox["selected_cells_in_ecoregion"],
                bbox["cell_selection_probability"],
                bbox["xmin"],
                bbox["ymin"],
                bbox["xmax"],
                bbox["ymax"],
            )
            for bbox in boxes
        ],
    )


def fetch_raw(con, config, from_aws: bool = False) -> None:
    """s3 -> local, unless already exists"""
    replace_boxes(con, config["bboxes"])
    if (
        not from_aws
        and con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'raw_buildings'").fetchone()[0]
    ):
        return

    src_url = f"s3://overturemaps-us-west-2/release/{config['release']}/theme={config['theme']}/type={config['type']}/*.parquet"
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_buildings AS
        WITH filtered AS (
            SELECT * FROM read_parquet('{src_url}')
            WHERE {bbox_filter(config["bboxes"])}
        )
        SELECT f.*, b.name AS box_name
        FROM filtered f
        JOIN boxes b
          ON f.bbox.xmin >= b.xmin AND f.bbox.xmax <= b.xmax
         AND f.bbox.ymin >= b.ymin AND f.bbox.ymax <= b.ymax
    """)


def build_footprints(con, config) -> None:
    con.execute("""
    CREATE OR REPLACE TABLE footprints AS
    SELECT
        r.id AS building_id,
        r.box_name,
        b.grid_x,
        b.grid_y,
        b.ecoregion_code,
        b.ecoregion_name,
        b.state,
        b.candidate_cells_in_ecoregion,
        b.selected_cells_in_ecoregion,
        b.cell_selection_probability,
        r.geometry,
        ST_Area(ST_Transform(r.geometry, 'OGC:CRS84', 'EPSG:5070')) AS footprint_m2_area,
        ST_X(ST_PointOnSurface(r.geometry)) AS anchor_lon,
        ST_Y(ST_PointOnSurface(r.geometry)) AS anchor_lat,
        ST_IsValid(r.geometry) AS is_valid,
        ST_GeometryType(r.geometry) as geometry_type,
        to_json(r.sources) AS provenance
    FROM raw_buildings r
    JOIN boxes b ON r.box_name = b.name
    WHERE r.is_underground IS NOT TRUE;
    """)


def write_geoparquet(con, out: Path) -> Path:
    temporary = out.with_name(f".{out.name}.tmp")
    temporary.unlink(missing_ok=True)
    con.execute(f"COPY footprints TO '{temporary}' (FORMAT PARQUET)")
    os.replace(temporary, out)
    return out


def extract(config: dict) -> Path:
    cache = REPO_ROOT / Path(config["cache"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    con = connect(cache)
    try:
        fetch_raw(con, config)
        build_footprints(con, config)
        out = REPO_ROOT / Path(config["out"])
        return write_geoparquet(con, out)
    finally:
        con.close()
