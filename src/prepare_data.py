"""Acquire and normalize the immutable SYNERGY v1.0 benchmark (CC0)."""
import argparse, concurrent.futures, csv, gzip, hashlib, io, json, urllib.request, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def fetch(entry):
    topic=entry["directoryLabel"].split("/")[-1]
    name=entry["label"]
    out=ROOT/"data"/"raw"/topic/name
    out.parent.mkdir(parents=True,exist_ok=True)
    expected=entry["dataFile"]["checksum"]["value"]
    if not out.exists() or hashlib.sha1(out.read_bytes()).hexdigest()!=expected:
        data=urllib.request.urlopen("https://dataverse.nl/api/access/datafile/"+str(entry["dataFile"]["id"]),timeout=180).read()
        assert hashlib.sha1(data).hexdigest()==expected, f"Checksum mismatch: {out}"
        out.write_bytes(data)
    return topic,name,out.stat().st_size

def abstract(inv):
    if not inv:return ""
    words=[(pos,word) for word,positions in inv.items() for pos in positions]
    return " ".join(w for _,w in sorted(words))

def normalize(topic):
    folder=ROOT/"data/raw"/topic
    labels=list(csv.DictReader((folder/"labels.csv").open()))
    works={}
    for zpath in sorted(folder.glob("works_*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for f in z.namelist():
                if f.endswith(".json"):
                    for r in json.loads(z.read(f)):
                        works[r["id"]]={"title":r.get("title") or "", "abstract":abstract(r.get("abstract_inverted_index")),"year":r.get("publication_year")}
    rows=[];seen={};conflicts=[]
    for r in labels:
        key=r["openalex_id"]
        if key not in works:continue
        y=int(r["label_included"])
        if key in seen:
            if seen[key]!=y: conflicts.append(key)
            continue
        seen[key]=y
        rows.append(dict(id=key,label=y,**works[key]))
    if conflicts:raise ValueError((topic,conflicts))
    rows.sort(key=lambda r:r["id"])
    out=ROOT/"data/processed"/(topic+".jsonl.gz")
    out.parent.mkdir(exist_ok=True,parents=True)
    payload="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows).encode()
    out.write_bytes(gzip.compress(payload,mtime=0))
    return {"topic":topic,"n":len(rows),"relevant":sum(r["label"] for r in rows),"missing_abstract":sum(not r["abstract"] for r in rows),"source_label_rows":len(labels),"sha256":hashlib.sha256(out.read_bytes()).hexdigest()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=6);args=parser.parse_args()
    meta=ROOT/"data/source_manifest.json"
    entries=json.loads(meta.read_text())["data"]["latestVersion"]["files"]
    entries=[x for x in entries if x["label"]=="labels.csv" or x["label"].startswith("works_") or x["label"].startswith("CITATION")]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(fetch,entries): print("download",*result,flush=True)
    topics=sorted(set(e["directoryLabel"].split("/")[-1] for e in entries))
    with concurrent.futures.ProcessPoolExecutor(min(args.workers,4)) as pool:
        info=list(pool.map(normalize,topics))
    (ROOT/"data/datasets.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2),flush=True)
if __name__=="__main__":main()
