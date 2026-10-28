"""
Using one 420 MB public parquet file, the first rows are downloaded until "--megabytes" of text are collected. 
The goal is to pass the result to preprocess_bert_pretrain and/or preprocess_bart_pretrain with --wikipedia <outdir>/source.

Requirement: 
    Need huggingface-hub and pyarrow. 
    (huggingface is a bitt quicker than the Wikipedia Dumps, for a small subset of data)
"""

import argparse
import os
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SOURCE = 'datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet'


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--outdir', required=True)
  parser.add_argument('--megabytes', type=float, default=55,
                      help='amount of article text to write (MB of UTF-8)')
  parser.add_argument('--num-shards', type=int, default=16)
  args = parser.parse_args()

  source_dir = os.path.join(args.outdir, 'source', 'en')
  os.makedirs(source_dir, exist_ok=True)
  target = int(args.megabytes * 1e6)

  start = time.time()
  articles, total = [], 0
  with HfFileSystem().open(SOURCE, 'rb') as f:
    for batch in pq.ParquetFile(f).iter_batches(batch_size=500, columns=['id', 'text']):
      for doc_id, text in zip(batch.column('id').to_pylist(), batch.column('text').to_pylist()):
        text = ' '.join(line.strip() for line in text.splitlines() if line.strip())
        if not text:
          continue
        line = 'wiki-{} {}'.format(doc_id, text)
        articles.append(line)
        total += len(line.encode('utf-8')) + 1
        if total >= target:
          break
      if total >= target:
        break

  for shard in range(args.num_shards):
    path = os.path.join(source_dir, '{}.txt'.format(shard))
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
      f.write('\n'.join(articles[shard::args.num_shards]))
  print('{} articles, {:.1f} MB of text, {} shards in {} ({:.0f} s)'.format(
      len(articles), total / 1e6, args.num_shards, source_dir, time.time() - start))


if __name__ == '__main__':
  main()
