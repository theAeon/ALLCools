import pathlib
import subprocess

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, as_completed

import pandas as pd
import xarray as xr
import zarr
import zarr.convenience
import zarr.creation
import zarr.hierarchy
import zarr.storage
from numcodecs import blosc

from .dataset import _determine_datasets, _count_single_zarr

from .._doc import *


@doc_params(
    generate_dataset_doc=generate_dataset_doc,
    allc_table_doc=allc_table_doc,
    chrom_size_path_doc=chrom_size_path_doc,
    regions_doc=generate_dataset_regions_doc,
    quantifiers_doc=generate_dataset_quantifiers_doc,
    obs_dim_doc=generate_dataset_obs_dim_doc,
    cpu_basic_doc=cpu_basic_doc,
    chunk_size_doc=generate_dataset_chunk_size_doc,
)
def generate_dataset_mpi(
    allc_table, output_path, regions, quantifiers, chrom_size_path, obs_dim="cell", cpu=1, chunk_size=None, tmpdir = None
):
    """\
    {generate_dataset_doc}

    Parameters
    ----------
    allc_table
        {allc_table_doc}
    output_path
        Output path of the MCDS dataset
    regions
        {regions_doc}
    quantifiers
        {quantifiers_doc}
    chrom_size_path
        {chrom_size_path_doc}
    obs_dim
        {obs_dim_doc}
    cpu
        {cpu_basic_doc}
    chunk_size
        {chunk_size_doc}

    Returns
    -------
    output_path
    """
    if isinstance(allc_table, (str, pathlib.Path)):
        allc_table = pd.read_csv(allc_table, sep="\t", header=None, index_col=0).squeeze()
        allc_table.index.name = obs_dim

    # determine index length and str dtype
    max_length = allc_table.index.map(lambda idx: len(idx)).max()
    obs_dim_dtype = f"<U{max_length}"

    # determine parallel chunk size
    n_sample = allc_table.size
    if chunk_size is None:
        chunk_size = min(n_sample, 50)

    # prepare regions and determine quantifiers
    pathlib.Path(output_path).mkdir(exist_ok=True)
    z = zarr.storage.DirectoryStore(path=output_path)
    root = zarr.hierarchy.group(store=z, overwrite=True)
    datasets, tmpdir = _determine_datasets(regions, quantifiers, chrom_size_path, tmpdir = tmpdir)
    # copy chrom_size_path to output_path
    subprocess.run(["cp", "-f", chrom_size_path, f"{output_path}/chrom_sizes.txt"], check=True)
    rgs = {}
    for region_dim, region_config in datasets.items():
        rgs[region_dim] = root.create_group(region_dim)
        # save region coords to the ds
        bed = pd.read_csv(f"{tmpdir}/{region_dim}.regions.csv", index_col=0)
        bed.columns = [f"{region_dim}_chrom", f"{region_dim}_start", f"{region_dim}_end"]
        bed.index.name = region_dim
        region_size = bed.index.size
        # append region bed to the saved ds
        ds = xr.Dataset()
        for col, data in bed.items():
            ds.coords[col] = data
        ds.coords[region_dim] = bed.index.values
        # change object dtype to string
        for k in ds.coords.keys():
            if ds.coords[k].dtype == "O":
                ds.coords[k] = ds.coords[k].astype(str)
        ds.to_zarr(f"{output_path}/{region_dim}", mode="w", consolidated=False)
        dsobs = rgs[region_dim].empty(
            name=obs_dim, shape=allc_table.index.size, chunks=(chunk_size), dtype=f"<U{max_length}"
        )
        dsobs.attrs["_ARRAY_DIMENSIONS"] = [obs_dim]
        count_mc_types = []
        for quant in region_config["quant"]:
            if quant.quant_type == "count":
                count_mc_types += quant.mc_types
        count_mc_types = list(set(count_mc_types))
        if len(count_mc_types) > 0:
            DA = rgs[region_dim].empty(
                name=f"{region_dim}_da",
                shape=(n_sample, region_size, len(count_mc_types), 2),
                chunks=(chunk_size, region_size, len(count_mc_types), 2),
                dtype="uint32",
            )
            DA.attrs["_ARRAY_DIMENSIONS"] = [obs_dim, region_dim, "mc_type", "count_type"]
            count = rgs[region_dim].array(name="count_type", data=(["mc", "cov"]), dtype="<U3")
            count.attrs["_ARRAY_DIMENSIONS"] = ["count_type"]
            mc = rgs[region_dim].array(name="mc_type", data=count_mc_types, dtype="<U3")
            mc.attrs["_ARRAY_DIMENSIONS"] = ["mc_type"]
        # deal with hypo-score, hyper-score quantifiers
        for quant in region_config["quant"]:
            if quant.quant_type == "hypo-score":
                for mc_type in quant.mc_types:
                    hypo = rgs[region_dim].empty(
                        name=f"{region_dim}_da_{mc_type}-hypo-score",
                        shape=(allc_table.size, region_size),
                        chunks=(chunk_size, region_size),
                        dtype="float16",
                    )
                    hypo.attrs["_ARRAY_DIMENSIONS"] = [obs_dim, region_dim]
            elif quant.quant_type == "hyper-score":
                for mc_type in quant.mc_types:
                    hyper = rgs[region_dim].empty(
                        name=f"{region_dim}_da_{mc_type}-hyper-score",
                        shape=(allc_table.size, region_size),
                        chunks=(chunk_size, region_size),
                        dtype="float16",
                    )
                    hyper.attrs["_ARRAY_DIMENSIONS"] = [obs_dim, region_dim]
    blosc.use_threads = False
    with MPIPoolExecutor(cpu, main = False) as exe:
        futures = {}
        # parallel on allc chunks and region_sets levels
        for i, chunk_start in enumerate(range(0, n_sample, chunk_size)):
            allc_chunk = allc_table[chunk_start : chunk_start + chunk_size]
            for region_dim, region_config in datasets.items():
                f = exe.submit(
                    _count_single_zarr,
                    allc_table=allc_chunk,
                    region_config=region_config,
                    obs_dim=obs_dim,
                    obs_dim_dtype=obs_dim_dtype,
                    region_dim=region_dim,
                    chunk_start=chunk_start,
                    regiongroup=rgs[region_dim],
                )
                futures[f] = (region_dim, i)
        for f in as_completed(futures):
            region_dim, i = futures[f]
            print(f"Chunk {i} of {region_dim} returned")
    blosc.use_threads = None
    from ..mcds.utilities import update_dataset_config

    update_dataset_config(
        output_path,
        config={
            "region_dim": None,
            "ds_region_dim": {region_dim: region_dim for region_dim in datasets.keys()},
            "ds_sample_dim": {region_dim: obs_dim for region_dim in datasets.keys()},
        },
    )
    for region_dim in datasets.keys():
        zarr.convenience.consolidate_metadata(f"{output_path}/{region_dim}")
    return output_path
