import duckdb

from thesis.data.overture import fetch_raw, get_config, replace_boxes


def test_frozen_bbox_config_is_ingestible():
    config = get_config()
    con = duckdb.connect(":memory:")

    replace_boxes(con, config["bboxes"])

    count, states, ecoregions = con.execute(
        "SELECT count(*), count(DISTINCT state), count(DISTINCT ecoregion_code) FROM boxes"
    ).fetchone()
    assert (count, states, ecoregions) == (120, 43, 10)


def test_cached_fetch_refreshes_bbox_metadata_without_remote_read():
    config = get_config()
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE raw_buildings (id VARCHAR)")

    fetch_raw(con, config)

    assert con.execute("SELECT count(*) FROM boxes").fetchone()[0] == 120
