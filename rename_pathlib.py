from pathlib import Path
### INPUTS
PATH = "../../torrent/downloads/"
### END INPUTS

p = Path(PATH)
torrents = [f for f in p.iterdir() if f.is_file()]
folders = [f for f in p.iterdir() if f.is_dir()]
file2folder = {f:i//20+1 for i,f in enumerate(torrents)}

for f,i in file2folder.items():
    to_rename = f.parent / f"{i}" / f.name
    f.rename(to_rename)
    print(f"{f} is renamed to {to_rename}")



