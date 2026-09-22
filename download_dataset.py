#!/usr/bin/env python3
"""Download a full Pokemon dataset (official artwork, sprites, metadata) from PokeAPI."""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SPRITES_RAW = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"
API = "https://pokeapi.co/api/v2"
MAX_DEX_ID = 1025
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pokegen-dataset/1.0"})


def get_json(url, retries=5):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt + random.random())
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"failed: {url}")


def download_file(url, dest, retries=5):
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 404:
                return "missing"
            r.raise_for_status()
            tmp.write_bytes(r.content)
            tmp.rename(dest)
            return "ok"
        except requests.RequestException:
            if attempt == retries - 1:
                return "failed"
            time.sleep(2 ** attempt + random.random())
    return "failed"


def list_pokemon():
    data = get_json(f"{API}/pokemon?limit=2000")
    out = []
    for entry in data["results"]:
        pid = int(entry["url"].rstrip("/").split("/")[-1])
        if pid <= MAX_DEX_ID:
            out.append({"id": pid, "name": entry["name"]})
    return sorted(out, key=lambda p: p["id"])


def fetch_metadata(entry):
    d = get_json(f"{API}/pokemon/{entry['id']}")
    return {
        "id": d["id"],
        "name": d["name"],
        "types": [t["type"]["name"] for t in d["types"]],
        "height": d["height"],
        "weight": d["weight"],
        "base_stats": {s["stat"]["name"]: s["base_stat"] for s in d["stats"]},
        "abilities": [a["ability"]["name"] for a in d["abilities"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--pixel-sprites", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.out)
    (root / "artwork").mkdir(parents=True, exist_ok=True)
    if args.pixel_sprites:
        (root / "sprites").mkdir(parents=True, exist_ok=True)

    print("Fetching Pokemon list from PokeAPI...")
    entries = list_pokemon()
    print(f"Found {len(entries)} Pokemon (dex 1-{MAX_DEX_ID})")

    image_jobs = []
    for e in entries:
        stem = f"{e['id']:04d}-{e['name']}"
        image_jobs.append((f"{SPRITES_RAW}/other/official-artwork/{e['id']}.png", root / "artwork" / f"{stem}.png"))
        if args.pixel_sprites:
            image_jobs.append((f"{SPRITES_RAW}/{e['id']}.png", root / "sprites" / f"{stem}.png"))

    print(f"Downloading {len(image_jobs)} images with {args.workers} workers...")
    counts = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_file, url, dest): (url, dest) for url, dest in image_jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            counts[status] += 1
            if status in ("failed", "missing"):
                print(f"  [{status}] {futures[fut][0]}")
            if i % 200 == 0:
                print(f"  {i}/{len(image_jobs)} images done")
    print("Image results:", counts)

    if not args.no_metadata:
        print("Fetching metadata (types/stats/abilities) from PokeAPI...")
        meta = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch_metadata, e): e for e in entries}
            for i, fut in enumerate(as_completed(futures), 1):
                e = futures[fut]
                try:
                    meta.append(fut.result())
                except Exception as err:
                    print(f"  [metadata failed] {e['name']}: {err}")
                    meta.append(e)
                if i % 200 == 0:
                    print(f"  {i}/{len(entries)} metadata done")
        meta.sort(key=lambda m: m["id"])
        (root / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"Wrote {root/'metadata.json'} ({len(meta)} entries)")

    artwork = list((root / "artwork").glob("*.png"))
    sprites = list((root / "sprites").glob("*.png")) if args.pixel_sprites else []
    print(f"Final dataset: {len(artwork)} artwork, {len(sprites)} sprites in {root.resolve()}")
    if len(artwork) < len(entries):
        print(f"WARNING: artwork count {len(artwork)} < expected {len(entries)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
