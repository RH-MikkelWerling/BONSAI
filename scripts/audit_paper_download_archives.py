"""Read-only, recursive ZIP schema/content-type check before paper downloads.

Detects common patient-level exports, including inside nested ZIPs and gzipped
tables. Does not certify anonymisation, inspect image pixels, or assess small
cell/differencing disclosure. Never prints data rows or identifier values.
"""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

IDENTIFIERS = {'patient_id', 'patient_ids', 'subject_id', 'subject_ids', 'person_id',
               'person_ids', 'pid', 'cpr', 'pnr', 'encrypted_cpr', 'hashed_patient_id',
               'date_of_birth', 'birth_date', 'patient_rows', 'patient_coordinates',
               'first_name', 'last_name', 'full_name', 'email', 'phone_number', 'address'}
PRIVATE = ('internal', 'server_only', 'direct_cr_predictions', 'patient_metadata',
           'patient_attributions', 'token_attributions', 'encoder_coordinates', 'subject_data')
RAW_SUFFIXES = {'.parquet', '.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.npy', '.npz', '.arrow'}


def audit_archive(path):
    result = {'archive': str(path), 'files_checked': 0, 'tables_checked': 0,
              'figures_requiring_visual_review': 0, 'issues': [],
              'scope': 'Automated schema/type check only; no guarantee of anonymity or small-cell disclosure protection'}

    def issue(member, reason):
        result['issues'].append({'member': member, 'reason': reason})

    def inspect_json(value, member):
        if isinstance(value, dict):
            if {str(key).lower() for key in value} & IDENTIFIERS:
                issue(member, 'Identifier/patient-level key in JSON; review inside secure environment')
            for child in value.values():
                inspect_json(child, member)
        elif isinstance(value, list):
            for child in value:
                inspect_json(child, member)

    def inspect_zip(source, prefix='', depth=0):
        if depth > 6:
            issue(prefix, 'Archive nesting limit exceeded')
            return
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                member = prefix + item.filename
                name = PurePosixPath(item.filename)
                lower = item.filename.lower()
                result['files_checked'] += 1
                if name.is_absolute() or '..' in name.parts:
                    issue(member, 'Unsafe archive path')
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    issue(member, 'Symlink in archive')
                if any(fragment in lower for fragment in PRIVATE) or name.suffix.lower() in RAW_SUFFIXES:
                    issue(member, 'Internal/patient-level artifact name or raw-data format')
                if item.flag_bits & 1:
                    issue(member, 'Encrypted member cannot be inspected')
                    continue
                with archive.open(item) as stream:
                    if lower.endswith('.zip'):
                        # Nested ZIPs require random access; release each after inspection.
                        inspect_zip(io.BytesIO(stream.read()), member + '!', depth + 1)
                    elif lower.endswith(('.csv', '.csv.gz')):
                        binary = gzip.GzipFile(fileobj=stream) if lower.endswith('.gz') else stream
                        text = io.TextIOWrapper(binary, encoding='utf-8-sig')
                        header = next(csv.reader(text), [])
                        result['tables_checked'] += 1
                        if {column.strip().lower() for column in header} & IDENTIFIERS:
                            issue(member, 'Identifier/patient-level table column')
                    elif lower.endswith('.json'):
                        if item.file_size > 64 * 1024 * 1024:
                            issue(member, 'JSON too large for automatic inspection')
                        else:
                            inspect_json(json.load(stream), member)
                    elif name.suffix.lower() in {'.png', '.pdf'}:
                        result['figures_requiring_visual_review'] += 1
                    elif name.suffix.lower() not in {'.tex', '.txt', '.md', '.sha256'}:
                        issue(member, 'Unrecognised export format; inspect before transfer')

    try:
        inspect_zip(path)
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as error:
        issue('', 'Archive/schema inspection failed: ' + type(error).__name__)
    result['automated_check_passed'] = not result['issues']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [audit_archive(path) for path in args.archives]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))
    for report in reports:
        print(('PASS automated checks: ' if report['automated_check_passed'] else 'REVIEW REQUIRED: ') + report['archive'])
    print('Report:', args.output)
    print('No identifier values exported. Passing is not a certification of anonymity; figures and disclosure risk still require review.')
    if any(not report['automated_check_passed'] for report in reports):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
