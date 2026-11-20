"""List the HDF5 structure of a DANDI NWB asset by streaming only metadata (no bulk download)."""
import sys

import h5py
import remfile
import requests

DANDISET, VERSION = sys.argv[1], sys.argv[2]
index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
listing = requests.get(f'https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/',
                       params={'page_size': 300}, timeout=60).json()['results']
asset = sorted(listing, key=lambda a: a['size'])[index]
print('asset', asset['path'], round(asset['size'] / 1e9, 2), 'GB', asset['asset_id'])
url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
with h5py.File(remfile.File(url), 'r') as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            size_mb = obj.size * obj.dtype.itemsize / 1e6
            print(f'{name}  shape={obj.shape} dtype={obj.dtype} {size_mb:.2f}MB')
    f.visititems(show)
