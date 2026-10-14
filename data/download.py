"""Download the public ACTG and writing data into data/raw/ and check provider-file SHA-256 sums.

usage: python data/download.py            (downloads ACTG 175 and the Noy-Zhang files, then checks all files)
       python data/download.py --check    (only checks the files already in data/raw/)
The Gerber-Green-Larimer file is distributed under the terms of use of the Yale ISPS Data Archive and must
be downloaded by hand (see data/README.md).
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / 'raw'
FILES = {
    'actg175/actg175.csv': ('https://archive.ics.uci.edu/static/public/890/data.csv',
                            'cf58638c7687764ccb32d037938142518293d505eec349ccec406d4d42ba9532'),
    'noy/fullsurvey.csv': ('https://osf.io/download/dch8w/',
                           '06f430ba9a9194039530721d68ad37f95552977a1ae1868c9228d1ce6334db0f'),
    'noy/grading_long_complete.csv': ('https://osf.io/download/hdszk/',
                                      '92771e43f2116df81a8681d281a94e94dec8ca4a2d0a04ff0a1ecfe3bcdc2779'),
    'noy/grading.xlsx': ('https://osf.io/download/m35kf/',
                         '76e66cb815a3e9b7e333ff50f8f49fd661857bc4614fae62fb987d06590edd1e'),
    'noy/grading_retry.xlsx': ('https://osf.io/download/3qcfb/',
                               '09009bc52b50d76e4585a8072a63115c819dbd21ab5da553c7f5bd01b2898e27'),
    'ggl/ggl_individuals.dta': (None, 'd8dd711c5216443d21e60050d768a7269b3ea9d656dbedd6802a738c2d0b080f'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='only check the files that are present')
    a = ap.parse_args()
    ok = True
    for name, (url, digest) in FILES.items():
        path = RAW / name
        if not path.exists() and url and not a.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f'downloading {name}', flush=True)
            urllib.request.urlretrieve(url, path)
        if not path.exists():
            print(f'missing   {name}' + ('' if url else '  (download by hand, see data/README.md)'))
            ok = False
            continue
        good = sha256(path) == digest
        ok &= good
        print(f'{"ok" if good else "MISMATCH":9s} {name}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
