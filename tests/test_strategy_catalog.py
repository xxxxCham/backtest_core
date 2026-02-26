from __future__ import annotations

from catalog.strategy_catalog import (
    build_entry_id,
    compute_params_hash,
    list_entries,
    move_entries,
    read_catalog,
    upsert_entry,
    write_catalog,
)


def test_catalog_roundtrip(tmp_path):
    path = tmp_path / "strategy_catalog.json"
    payload = {"schema_version": 1, "entries": []}
    write_catalog(payload, path=path)
    loaded = read_catalog(path=path)
    assert loaded["schema_version"] == 1
    assert loaded["entries"] == []


def test_upsert_and_filters(tmp_path):
    path = tmp_path / "strategy_catalog.json"
    params_hash = compute_params_hash({"fast": 10})
    entry_id = build_entry_id("ema_cross", "BTCUSDC", "1h", params_hash)
    entry = {
        "id": entry_id,
        "strategy_name": "ema_cross",
        "symbol": "BTCUSDC",
        "timeframe": "1h",
        "params_hash": params_hash,
        "category": "p1_builder_inbox",
        "status": "active",
        "tags": ["builder_out"],
    }
    upsert_entry(entry, path=path)

    entries = list_entries(path=path, categories=["p1_builder_inbox"])
    assert len(entries) == 1
    assert entries[0]["id"] == entry_id

    moved = move_entries([entry_id], "p3_watchlist", path=path)
    assert moved == 1
    entries = list_entries(path=path, categories=["p3_watchlist"])
    assert len(entries) == 1
