"""Development offered-route study; final generation requires prior manifest approval."""

import argparse
import hashlib
from pathlib import Path

from wnp.offered_routes import OfferedRouteConfig, run_development, run_final, seal_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('development','final'),default='development')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--approved-manifest',type=Path)
    parser.add_argument('--approved-hash')
    parser.add_argument('--seal-manifest',action='store_true',help='Seal after successful development; never run final layouts')
    args=parser.parse_args()
    config=OfferedRouteConfig()
    root=Path(__file__).resolve().parents[1]
    output=args.output or root/'results'/('offered_routes_tradeoffs_dev' if args.stage=='development' else 'offered_routes_tradeoffs_final')
    if args.stage=='development':
        if args.approved_manifest or args.approved_hash:
            parser.error('approval parameters apply only to the final stage')
        report=run_development(config,output)
        if args.seal_manifest and report['status']=='passed':
            seal_manifest(config,output)
            path=output/'final_manifest.json'
            print(f'Sealed final manifest SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}')
    else:
        if args.seal_manifest:
            parser.error('final execution cannot seal its own manifest')
        report=run_final(config,output,args.approved_manifest,args.approved_hash)
    print(f'{report["status"]}: {report["preflight_passed_layouts"]} preflight passes, '
          f'{report["preflight_failed_layouts"]} failures; outputs {output}')
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':
    raise SystemExit(main())
