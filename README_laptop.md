## preprocessing on laptop cpu's

Python 3.12+ and git

```
mkdir lddl-demo
cd lddl-demo
git clone -b update-dask-env https://github.com/ofk123/LDDL
python -m venv lddl-env
lddl-env\Scripts\activate 
pip install -r LDDL/requirements.txt
pip install --no-deps ./LDDL
```

`source lddl-env/bin/activate` on Mac

and then if the installs succeed, run the preprocessing like this
```
python LDDL/make_smaller_text.py --outdir data/wikipedia --megabytes 40
preprocess_bert_pretrain --schedule local --local-n-workers 4 --wikipedia data/wikipedia/source --sink out/bert --num-blocks 64
```

NB: The traceback after Running the dask pipeline took ... s  should be harmless. I see it on my end too.
 
dask:

While preprocess_bert_pretrain runs, open the dashboard: http://localhost:8787/status
