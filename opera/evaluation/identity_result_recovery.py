"""Recover scalar identity contrasts from completed trusted worker results.

Run before loading the large prediction table. Only one saved batch is loaded
at a time; the final table stays on disk. No statistical computation is changed.
"""
import argparse
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd

from opera.evaluation.distributed_analysis import load_pickle, save_json


class CSVResultTable:
    """Final-output table backed by a CSV, rather than an in-memory DataFrame."""
    def __init__(self, path, rows):
        self.path, self.rows = Path(path), rows

    def __len__(self):
        return self.rows

    def to_csv(self, path, index=False, chunksize=None):
        if index:
            raise ValueError('Recovered tables already omit the index')
        shutil.copyfile(self.path, path)


def completed_identity_sources(directory):
    directory = Path(directory)
    queue = directory / 'queue.json'
    if not queue.exists():
        return None
    total = json.loads(queue.read_text())['total']
    paths = [directory / f'batch_{n:05d}.result_INTERNAL.pkl' for n in range(total)]
    if not paths or not all(p.is_file() for p in paths):
        return None
    if list(directory.glob('batch_*.failed.json')):
        raise ValueError('Identity queue has failed batches; inspect before recovery')
    return paths


def recover(directory):
    directory = Path(directory)
    sources = completed_identity_sources(directory)
    if sources is None:
        raise ValueError('Every identity batch must be complete before recovery')
    identities = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sources}
    output = directory / 'scalar_contrasts_INTERNAL.csv'
    marker = directory / 'scalar_contrasts_complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved['sources'] != identities or not output.exists()
                or output.stat().st_size != saved['output_bytes']):
            raise ValueError('Recovered identity table or source results changed')
        return CSVResultTable(output, saved['rows'])
    temporary = directory / ('.scalar_' + uuid.uuid4().hex)
    rows, columns = 0, None
    try:
        for number, source in enumerate(sources):
            part = load_pickle(source)
            frame = pd.DataFrame(part[1])  # Only contrasts are consumed by identity sensitivity.
            if columns is None:
                columns = list(frame.columns)
            elif list(frame.columns) != columns:
                raise ValueError('Identity batch schemas differ')
            frame.to_csv(temporary, mode='a', header=number == 0, index=False)
            rows += len(frame)
            del frame, part
            if number % 25 == 0 or number + 1 == len(sources):
                print(f'Identity scalar recovery: {number+1}/{len(sources)} batches', flush=True)
        temporary.replace(output)
        save_json(marker, dict(sources=identities, rows=rows, output_bytes=output.stat().st_size,
                               scope='Exact batch contrast rows; other batch outputs not needed'))
    finally:
        temporary.unlink(missing_ok=True)
    return CSVResultTable(output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    recover(args.directory)


if __name__ == '__main__':
    main()
