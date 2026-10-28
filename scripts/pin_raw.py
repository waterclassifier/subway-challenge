"""Re-pin one source in data/raw/MANIFEST.tsv to another upstream commit.

For every manifest row of the source, the file is downloaded from the new
commit and the row's source_commit, fetched date and sha256 are rewritten.
Rows of other sources are left untouched. The raw files themselves are not
written: run snakemake afterwards, which sees the changed pins, refetches the
source and rebuilds everything downstream of it.

Examples:
    python scripts/pin_raw.py --source beijing-subway-tools --commit latest
    python scripts/pin_raw.py --source beijing-subway-schedule --commit 2d08831cfe89d5372f5d10491cd691e4f43ab76f
"""

import argparse
import csv
import datetime
import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_raw import download, raw_url  # noqa: E402


def resolve_commit(repo_url, ref):
    """Turn a full commit sha, branch, tag or "latest" into a full commit sha."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    ref = "HEAD" if ref == "latest" else ref
    out = subprocess.run(
        ["git", "ls-remote", repo_url, ref, f"refs/tags/{ref}^{{}}"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    if not out:
        raise SystemExit(f"{ref!r} is not a branch or tag of {repo_url}; "
                         f"give \"latest\", a branch, a tag, or a full 40-character commit sha")
    # A peeled annotated tag (^{}) points at the commit itself, so prefer it.
    shas = dict(zip(out[1::2], out[0::2]))
    return shas.get(f"refs/tags/{ref}^{{}}", out[0])


def pin(manifest, source, ref):
    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        rows = list(reader)
    mine = [r for r in rows if r["path"].startswith(source + "/")]
    if not mine:
        raise SystemExit(f"no rows for {source!r} in {manifest}")
    repos = {r["source_repo"] for r in mine}
    if len(repos) != 1:
        raise SystemExit(f"{source!r} rows name more than one repo: {sorted(repos)}")
    commit = resolve_commit(repos.pop(), ref)
    today = datetime.date.today().isoformat()

    for row in mine:
        url = raw_url(row["source_repo"], commit, row["source_path"])
        try:
            data = download(url)
        except OSError as e:
            raise SystemExit(f"{row['source_path']} not found at {commit[:7]} ({e}); "
                             f"fix or remove its manifest row and retry") from None
        digest = hashlib.sha256(data).hexdigest()
        status = "unchanged" if digest == row["sha256"] else "changed"
        print(f"{row['path']}: {status}")
        row.update(source_commit=commit, fetched=today, sha256=digest)

    # Write only after every file downloaded, so a failure leaves the manifest as it was.
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{source} pinned to {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/raw/MANIFEST.tsv")
    parser.add_argument("--source", required=True, help="first path component, e.g. beijing-subway-tools")
    parser.add_argument("--commit", required=True, help='full commit sha, branch, tag, or "latest"')
    args = parser.parse_args()
    pin(args.manifest, args.source, args.commit)
