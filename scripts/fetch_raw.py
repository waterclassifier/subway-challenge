"""Download the pinned raw files of one source listed in data/raw/MANIFEST.tsv.

Each manifest row names a GitHub repo, a commit and a path in that repo. The
file is downloaded from that exact commit, checked against the recorded
sha256, and only then moved into place, so a finished run always leaves the
same bytes on disk.
"""

import argparse
import csv
import hashlib
import os
import tempfile
import time
import urllib.request
from pathlib import Path

RETRIES = 4


def raw_url(repo_url, commit, path):
    owner_repo = repo_url.removeprefix("https://github.com/").strip("/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{path}"


def download(url):
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except OSError:
            if attempt == RETRIES:
                raise
            time.sleep(2 ** (attempt + 1))


def fetch(manifest, source, root):
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["path"].startswith(source + "/")]
    if not rows:
        raise SystemExit(f"no rows for {source!r} in {manifest}")
    for row in rows:
        url = raw_url(row["source_repo"], row["source_commit"], row["source_path"])
        data = download(url)
        digest = hashlib.sha256(data).hexdigest()
        if digest != row["sha256"]:
            raise SystemExit(f"sha256 mismatch for {url}: got {digest}, expected {row['sha256']}")
        dest = Path(root) / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".fetch-")
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(tmp, dest)
        print(f"{row['path']}: ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", required=True, help="first path component, e.g. beijing-subway-schedule")
    parser.add_argument("--root", required=True, help="directory the manifest paths are relative to")
    args = parser.parse_args()
    fetch(args.manifest, args.source, args.root)
