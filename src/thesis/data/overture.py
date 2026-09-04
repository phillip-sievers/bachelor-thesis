from pathlib import Path

import duckdb as db
import yaml

CONF_PATH = Path("/Users/phillip/repositories/bachelor-thesis/configs/overture.yaml")


def get_config(path: Path = CONF_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
        ST_GeometryType(r.geometry) as geometry_type
    FROM raw_buildings r
    JOIN boxes b ON r.box_name = b.name
    WHERE r.is_underground IS NOT TRUE;
    """)


def write_geoparquet(con, out: Path) -> Path:
    con.execute(f"COPY footprints TO '{out}' (FORMAT PARQUET)")
    return out


def extract(config: dict) -> Path:
    cache = Path(config["cache"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    con = connect(cache)
    fetch_raw(con, config)
    build_footprints(con, config)
    return write_geoparquet(con, config["out"])


if __name__ == "__main__":
    out_file = extract(get_config())
    # df = gpd.read_parquet(out_file)

    config = get_config()
    con = connect(Path(config["cache"]))

    metrics = con.sql("""
        SELECT
            count(*) AS n,
            count(*) FILTER (WHERE NOT is_valid) as n_invalid,
            count(*) FILTER (WHERE geometry_type <> 'POLYGON') as n_nonpolygon
        FROM footprints
    """).df()

    print(metrics)
