#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

# to_parquet
import dask
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

# to_textfiles
import io
import uuid
from contextlib import ExitStack
from dask.highlevelgraph import HighLevelGraph
from dask.utils import ensure_unicode, ensure_bytes, system_encoding
from fsspec.core import open_files

#
# parquet files
#


def _write_partition_binned(pairs, path, partition_index, bin_size, nbins,
                            schema):
  fs, path = url_to_fs(path)
  bins = [[] for _ in range(nbins)]
  for pair in pairs:
    bin_id = (pair['num_tokens'] - 1) // bin_size
    bin_id = nbins - 1 if bin_id > nbins - 1 else bin_id
    bins[bin_id].append(pair)

  for bin_id, bin_pairs in enumerate(bins):
    columns = {
        name: [pair[name] for pair in bin_pairs]
        for name in schema.names
        if name != 'bin_id'
    }
    columns['bin_id'] = [bin_id] * len(bin_pairs)
    table = pa.Table.from_pydict(columns, schema=schema)
    filename = 'part.{}.parquet_{}'.format(partition_index, bin_id)
    with fs.open(fs.sep.join([path, filename]), 'wb') as f:
      pq.write_table(table, f)
  return [len(bin_pairs) for bin_pairs in bins]


def to_parquet_binned(pairs, path, bin_size, nbins, schema):
  """this writes a 'bag' of pairs as parquet files, one per partition and bin.

  Partition i of the bag is written to part.<i>.parquet_<bin>
  under path, where empty bins give empty files.
  """
  schema = pa.schema(schema)
  fs, fs_path = url_to_fs(path)
  fs.makedirs(fs_path, exist_ok=True)
  return dask.compute(*[
      dask.delayed(_write_partition_binned)(
          partition,
          path,
          partition_index,
          bin_size,
          nbins,
          schema,
      ) for partition_index, partition in enumerate(pairs.to_delayed())
  ])


#
# text files
#


class file_namer(object):

  def __init__(self, bin_size, nbins, prefix=""):
    self.__bin_size = bin_size
    self.__nbins = nbins
    self.__prefix = prefix

  def name_function(self, i):
    num = i // self.__nbins
    bin_val = i % self.__nbins
    return '%s%d_%d' % (self.__prefix, num, bin_val)


def _to_textfiles_chunk_binned(data, lazy_files, last_endline, bin_size):
  nbins = len(lazy_files)
  with ExitStack() as stack:
    fs = [stack.enter_context(lazy_file) for lazy_file in lazy_files]
    if isinstance(fs[0], io.TextIOWrapper):
      endline = "\n"
      ensure = ensure_unicode
    else:
      endline = b"\n"
      ensure = ensure_bytes
    starteds = [False] * nbins
    for d in data:
      # Assuming the last character containes the number of tokens.
      seq_len = int(d.split()[-1])
      bin_id = (seq_len - 1) // bin_size
      bin_id = nbins - 1 if bin_id > nbins - 1 else bin_id
      if starteds[bin_id]:
        fs[bin_id].write(endline)
      else:
        starteds[bin_id] = True
      fs[bin_id].write(ensure(d))
    if last_endline:
      for f in fs:
        f.write(endline)


def to_textfiles_binned(b,
                        path,
                        bin_size=64,
                        nbins=8,
                        compression="infer",
                        encoding=system_encoding,
                        compute=True,
                        storage_options=None,
                        last_endline=False,
                        **kwargs):

  mode = "wb" if encoding is None else "wt"
  files = open_files(path,
                     compression=compression,
                     mode=mode,
                     encoding=encoding,
                     name_function=file_namer(bin_size, nbins).name_function,
                     num=b.npartitions * nbins,
                     **(storage_options or {}))

  name = "to-textfiles-binned-" + uuid.uuid4().hex
  dsk = {(name, i): (_to_textfiles_chunk_binned, (b.name, i),
                     files[k:k + nbins], last_endline, bin_size)
         for i, k in enumerate(range(0, len(files), nbins))}
  graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
  out = type(b)(graph, name, b.npartitions)

  if compute:
    out.compute(**kwargs)
    return [f.path for f in files]
  else:
    return out.to_delayed()
