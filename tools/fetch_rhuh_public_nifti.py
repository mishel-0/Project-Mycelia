"""Fetch the CC BY 4.0 NIfTI image/mask subset used by the MYCELIA 3D pilot.

Only images/, masks/, index.csv, and README.md are downloaded. The source repo
revision is pinned and every large file is checked against its Hub SHA-256.
No DICOM, meshes, or visualization-only volumes are requested.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


REPO = 'AIOmarRehan/medtrace-rhuh-gbm-derived'
REVISION = '713d099a2ac6ccd3852f1ad9acdc3b5368c4df3a'
BASE = f'https://huggingface.co/datasets/{REPO}'


def _get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'MYCELIA-research-data-fetch/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def _digest_file(path, algorithm, git_blob_size=None):
    digest = hashlib.new(algorithm)
    if git_blob_size is not None:
        digest.update(f'blob {git_blob_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download(entry, destination):
    target = destination / entry['path']
    lfs_sha = entry.get('lfs', {}).get('oid')
    expected_sha = lfs_sha or entry.get('oid')
    expected_size = entry.get('size')
    if not expected_sha or not isinstance(expected_size, int):
        raise ValueError(f"Hub did not provide an integrity record for {entry['path']}")
    algorithm = 'sha256' if lfs_sha else 'sha1'
    git_blob_size = None if lfs_sha else expected_size
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target.is_file() and target.stat().st_size == expected_size and
            _digest_file(target, algorithm, git_blob_size) == expected_sha):
        return entry['path'], expected_size, 'cached'
    url = (f'{BASE}/resolve/{REVISION}/'
           f"{urllib.parse.quote(entry['path'], safe='/')}?download=true")
    request = urllib.request.Request(url, headers={'User-Agent': 'MYCELIA-research-data-fetch/1.0'})
    partial = target.with_name(target.name + '.part')
    digest = hashlib.new(algorithm)
    if git_blob_size is not None:
        digest.update(f'blob {git_blob_size}\0'.encode())
    count = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if count != expected_size or digest.hexdigest() != expected_sha:
            raise ValueError(f"integrity mismatch for {entry['path']}: {count} bytes, sha256={digest.hexdigest()}")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return entry['path'], count, 'downloaded'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='destination folder for the pinned derivative subset')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    args.output.mkdir(parents=True, exist_ok=True)
    info = _get_json(f'https://huggingface.co/api/datasets/{REPO}')
    license_id = info.get('cardData', {}).get('license')
    if license_id != 'cc-by-4.0':
        raise SystemExit(f'expected CC BY 4.0 repository metadata, received {license_id!r}')
    entries = _get_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&expand=false')
    selected = [item for item in entries if item.get('type') == 'file' and
                (item['path'] == 'index.csv' or item['path'] == 'README.md' or
                 item['path'].startswith(('images/', 'masks/')))]
    if len([item for item in selected if item['path'].startswith('images/')]) != 336:
        raise SystemExit('expected 336 indexed sequence image volumes')
    if len([item for item in selected if item['path'].startswith('masks/')]) != 336:
        raise SystemExit('expected 336 aligned mask volumes')
    total_bytes = sum(int(item.get('size', 0)) for item in selected)
    print(f'Pinned {REPO}@{REVISION}: {len(selected)} files, {total_bytes/1024**2:.1f} MiB, CC BY 4.0')
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download, item, args.output) for item in selected]
        for n, future in enumerate(as_completed(futures), 1):
            completed.append(future.result())
            if n % 40 == 0 or n == len(futures):
                print(f'{n}/{len(futures)} verified; {sum(row[1] for row in completed)/1024**2:.1f} MiB')
    manifest = {
        'repo': REPO, 'revision': REVISION, 'license': license_id,
        'file_count': len(completed), 'total_bytes': sum(row[1] for row in completed),
        'downloaded_files': sum(row[2] == 'downloaded' for row in completed),
        'cached_files': sum(row[2] == 'cached' for row in completed),
        'included_paths': sorted(row[0] for row in completed),
    }
    (args.output / 'mycelia-download-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'included_paths'}, indent=2))


if __name__ == '__main__':
    main()
