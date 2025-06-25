# [HotStorage '25] Unlocking the Unusable: A Proactive Caching Framework for Reusing Partial Overlapped Data

Mosaic-Cache, a proactive and general caching framework that enables applications with efficient partial overlapped data reuse through novel overlap-aware cache interfaces for fast content-level reuse.

Mosaic-Cache prototype is on top of [ADIOS2: The Adaptable Input Output System version 2](https://adios2.readthedocs.io/).

## 🚀 Deployment

To build ADIOS2 with the Mosaic-Cache enhancements, run:

```bash
source scripts/build_scripts/build-adios2-kvcache-spatial-index.sh --build
```
Then start the redis server and ADIOS 2 remote server:
```bash
./build/sw/redis-7.2.3/src/redis-server

# on local machine
./build/bin/adios2-remote-server -v

# port forward for remote access
ssh -L 26200:<Remote Address>:26200 -l <username> <Remote Address> "path_to_ADIOS2/build/bin/adios2_remote_server -v"
```
## 🧪 Testing

The cache behavior is controlled via environment variables. Below are different testing modes:

### 🔹 No Cache (Remote Access Only)

```bash
export DoRemote=1
```

### 🔹 KV-Cache (Traditional + Remote)

```bash
export DoRemote=1
export useKVCache=1
export useTraditionalCache=1
```

### 🔹 Meta-Match (Scanning and parsing all cache keys from strings for each query + Traditional + Remote)

```bash
export DoRemote=1
export useKVCache=1
export useFunctionalKVCache=1
```

### 🔹 Mosaic-Cache

```bash
export DoRemote=1
export useKVCache=1
export useFunctionalKVCacheWithMetaManager=1
```

## Python Testing

```bash
sudo apt install python3-dev
sudo apt install python3-pip
python3 -m pip install --user numpy mpi4py
export BUILD_DIR=/data/gc/ADIOS2/build
export SW_DIR=${BUILD_DIR}/sw
export PYTHONPATH=${BUILD_DIR}/lib/python3/dist-packages
export LD_LIBRARY_PATH=${SW_DIR}/hiredis/lib:$LD_LIBRARY_PATH
```
test package installation: python3 → import adios2

## Python Sample Script
```python3

    adios = adios2.ADIOS()
    queryIO = adios.DeclareIO("query")

    reader = queryIO.Open(data_file, adios2.Mode.ReadRandomAccess)
    var = queryIO.InquireVariable(var_name)

    query_df = pd.read_csv(queryFileName, header=0)
    query_df = query_df.sample(frac=1).reset_index(drop=True)
    # start time
    global_start = time.time()
    for index, row in query_df.iterrows():
        start3D = eval(row['start'])
        count3D = eval(row['count'])
        print(f"Processing query {index}, with start3D={start3D} count3D={count3D}")
        
        query_start = time.time()
        var.SetSelection([start3D, count3D])
        res = np.zeros(count3D, dtype=np.double)
        reader.Get(var, res, adios2.Mode.Sync)
        query_end = time.time()
```
