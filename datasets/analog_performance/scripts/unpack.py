"""Unpack and SHA-256 verify selected dataset formats (standard library only)."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['sqlite', 'csv', 'jsonl', 'all'], default='sqlite')
    parser.add_argument('--verify-only', action='store_true', help='Check compressed and original hashes without writing output')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'assets.json').read_text(encoding='utf-8'))
    for asset in manifest['assets']:
        if args.format != 'all' and asset['format'] != args.format:
            continue
        compressed = (root / asset['compressed_path']).resolve()
        target = (root / asset['path']).resolve()
        if root not in compressed.parents or root not in target.parents:
            raise ValueError('Asset path is outside the dataset directory')
        if digest(compressed) != asset['compressed_sha256']:
            raise ValueError('Compressed checksum mismatch: ' + str(compressed))
        if target.exists() and not args.verify_only:
            if target.stat().st_size == asset['bytes'] and digest(target) == asset['sha256']:
                print('Already verified:', asset['path'])
                continue
            raise FileExistsError('Refusing to replace a different existing file: ' + str(target))
        temporary = target.with_name(target.name + '.partial')
        writer = None
        try:
            if not args.verify_only:
                writer = temporary.open('xb')
            h = hashlib.sha256()
            size = 0
            with gzip.open(compressed, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                    size += len(block)
                    if size > asset['bytes']:
                        raise ValueError('Uncompressed size exceeds manifest')
                    if writer:
                        writer.write(block)
            if size != asset['bytes'] or h.hexdigest() != asset['sha256']:
                raise ValueError('Uncompressed checksum mismatch: ' + str(compressed))
            if writer:
                writer.close()
                os.replace(temporary, target)
            print('Verified:', asset['path'])
        finally:
            if writer:
                writer.close()
                if temporary.exists():
                    temporary.unlink()


if __name__ == '__main__':
    main()
