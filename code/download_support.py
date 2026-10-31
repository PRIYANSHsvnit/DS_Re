from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
root=Path(__file__).resolve().parent.parent
target=root/'data/support2.csv'
if target.exists():
    raise SystemExit(f"Refusing to overwrite existing file: {target}")
url='https://hbiostat.org/data/repo/support2csv.zip'
with urlopen(url,timeout=90) as response: payload=response.read()
with ZipFile(BytesIO(payload)) as archive:
    names=[n for n in archive.namelist() if Path(n).name.lower()=='support2.csv']
    if len(names)!=1: raise SystemExit('Unexpected archive contents.')
    data=archive.read(names[0])
target.parent.mkdir(parents=True,exist_ok=True)
target.write_bytes(data)
print(f'Saved {target}; see DATA_PROVENANCE.md for required attribution.')
