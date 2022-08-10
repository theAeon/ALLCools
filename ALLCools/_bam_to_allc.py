"""
This file is modified from methylpy https://github.com/yupenghe/methylpy.

Author: Yupeng He

Notes added on 07/03/2022 - Difference between bismark and hisat-3n
Read mpileup doc first: http://www.htslib.org/doc/samtools-mpileup.html

For Bismark SE mapping:
Bismark converted the reads orientation based on their conversion type.
C to T conversion reads are always map to the forward strand, regardless of the R1 R2
G to A conversion reads are always map to the reverse strand, regardless of the R1 R2
Therefore, in the original bam-to-allc function, we simply consider the strandness in
the pileup format to distinguish which C needs to be counted.
There are two situations:
1. If the ref base is C, the read need to map to forward strand in order to be counted
[.ATCGN] corresponding to the forward strand
2. If the ref base is G, the read need to map to reverse strand in order to be counted
[,atcgn] corresponding to the reverse strand

For Hisat-3n PE or Biskarp PE mapping:
R1 R2 are mapped as their original orientation, therefore,
both the C to T and G to A conversion reads can have forward and reverse strand alignment.
We can not distinguish the conversion type by strand in input bam.
Here I add a check using the YZ tag of hisat-3n BAM file or XG tag of bismark BAM file.
If the YZ tag is "+" or XG tag is "CT", the read is C to T conversion, I change the flag to forward mapping
no matter R1 or R2 by read.is_forward = True
If the YZ tag is "-" or XG tag is "GA", the read is G to A conversion, I change the flag to reverse mapping
no matter R1 or R2 by read.is_forward = False
In this case, the read orientation is the same as bismark bam file, and the following base
count code no need to change.

"""

import collections
import logging
import pathlib
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pysam

from ._doc import *
from ._open import open_allc, open_bam
from .utilities import genome_region_chunks

# logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def _is_read_ct_conversion_hisat3n(read):
    return read.get_tag("YZ") == "+"


def _is_read_ct_conversion_bismark(read):
    return read.get_tag("XG") == "CT"


def _convert_bam_strandness(in_bam_path, out_bam_path):
    with pysam.AlignmentFile(in_bam_path) as in_bam, pysam.AlignmentFile(
        out_bam_path, header=in_bam.header, mode="wb"
    ) as out_bam:
        is_ct_func = None
        for read in in_bam:
            if is_ct_func is None:
                if read.has_tag("YZ"):
                    is_ct_func = _is_read_ct_conversion_hisat3n
                elif read.has_tag("XG"):
                    is_ct_func = _is_read_ct_conversion_bismark
                else:
                    raise ValueError(
                        "The bam file reads has no conversion type tag "
                        "(XG by bismark or YZ by hisat-3n). Please note that this function can "
                        "only process bam files generated by bismark or hisat-3n."
                    )
            if is_ct_func(read):
                read.is_forward = True
                if read.is_paired:
                    read.mate_is_forward = True
            else:
                read.is_forward = False
                if read.is_paired:
                    read.mate_is_forward = False
            out_bam.write(read)
    return


def _read_faidx(faidx_path):
    """
    Read fadix of reference fasta file.

    samtools fadix ref.fa
    """
    return pd.read_csv(
        faidx_path,
        index_col=0,
        header=None,
        sep="\t",
        names=["NAME", "LENGTH", "OFFSET", "LINEBASES", "LINEWIDTH"],
    )


def _get_chromosome_sequence_upper(fasta_path, fai_df, query_chrom):
    """Read a whole chromosome sequence into memory."""
    chrom_pointer = fai_df.loc[query_chrom, "OFFSET"]
    tail = fai_df.loc[query_chrom, "LINEBASES"] - fai_df.loc[query_chrom, "LINEWIDTH"]
    seq = ""
    with open(fasta_path) as f:
        f.seek(chrom_pointer)
        for line in f:
            if line[0] == ">":
                break
            seq += line[:tail]  # trim \n
    return seq.upper()


def _get_bam_chrom_index(bam_path):
    result = subprocess.run(["samtools", "idxstats", bam_path], stdout=subprocess.PIPE, encoding="utf8").stdout

    chrom_set = set()
    for line in result.split("\n"):
        chrom = line.split("\t")[0]
        if chrom not in ["", "*"]:
            chrom_set.add(chrom)
    return pd.Index(chrom_set)


def _bam_to_allc_worker(
    bam_path,
    reference_fasta,
    fai_df,
    output_path,
    region=None,
    num_upstr_bases=0,
    num_downstr_bases=2,
    buffer_line_number=100000,
    min_mapq=0,
    min_base_quality=1,
    compress_level=5,
    tabix=True,
    save_count_df=False,
):
    """None parallel bam_to_allc worker function, call by bam_to_allc."""
    # mpileup
    if region is None:
        mpileup_cmd = f"samtools mpileup -Q {min_base_quality} " f"-q {min_mapq} -B -f {reference_fasta} {bam_path}"
        pipes = subprocess.Popen(
            shlex.split(mpileup_cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    else:
        bam_handle = open_bam(
            bam_path,
            region=region,
            mode="r",
            include_header=True,
            samtools_parms_str=None,
        )
        mpileup_cmd = f"samtools mpileup -Q {min_base_quality} " f"-q {min_mapq} -B -f {reference_fasta} -"
        pipes = subprocess.Popen(
            shlex.split(mpileup_cmd),
            stdin=bam_handle.file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    result_handle = pipes.stdout

    output_file_handler = open_allc(output_path, mode="w", compresslevel=compress_level)

    # initialize variables
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    mc_sites = {"C", "G"}
    context_len = num_upstr_bases + 1 + num_downstr_bases
    cur_chrom = ""
    line_counts = 0
    total_line = 0
    out = ""
    seq = None  # whole cur_chrom seq
    chr_out_pos_list = []
    cur_out_pos = 0
    cov_dict = collections.defaultdict(int)  # context: cov_total
    mc_dict = collections.defaultdict(int)  # context: mc_total

    # process mpileup result
    for line in result_handle:
        total_line += 1
        fields = line.split("\t")
        fields[2] = fields[2].upper()
        # if chrom changed, read whole chrom seq from fasta
        if fields[0] != cur_chrom:
            cur_chrom = fields[0]
            chr_out_pos_list.append((cur_chrom, str(cur_out_pos)))
            # get seq for cur_chrom
            seq = _get_chromosome_sequence_upper(reference_fasta, fai_df, cur_chrom)

        if fields[2] not in mc_sites:
            continue

        # deal with indels
        read_bases = fields[4]
        incons_basecalls = read_bases.count("+") + read_bases.count("-")
        if incons_basecalls > 0:
            read_bases_no_indel = ""
            index = 0
            prev_index = 0
            while index < len(read_bases):
                if read_bases[index] == "+" or read_bases[index] == "-":
                    # get insert size
                    indel_size = ""
                    ind = index + 1
                    while True:
                        try:
                            int(read_bases[ind])
                            indel_size += read_bases[ind]
                            ind += 1
                        except Exception:
                            break
                    try:
                        # sometimes +/- does not follow by a number and
                        # it should be ignored
                        indel_size = int(indel_size)
                    except Exception:
                        index += 1
                        continue
                    read_bases_no_indel += read_bases[prev_index:index]
                    index = ind + indel_size
                    prev_index = index
                else:
                    index += 1
            read_bases_no_indel += read_bases[prev_index:index]
            fields[4] = read_bases_no_indel

        # count converted and unconverted bases
        if fields[2] == "C":
            # mpileup pos is 1-based, turn into 0 based
            pos = int(fields[1]) - 1
            try:
                context = seq[(pos - num_upstr_bases) : (pos + num_downstr_bases + 1)]
            except Exception:  # complete context is not available, skip
                continue
            unconverted_c = fields[4].count(".")
            converted_c = fields[4].count("T")
            cov = unconverted_c + converted_c
            if cov > 0 and len(context) == context_len:
                line_counts += 1
                data = (
                    "\t".join(
                        [
                            cur_chrom,
                            str(pos + 1),
                            "+",
                            context,
                            str(unconverted_c),
                            str(cov),
                            "1",
                        ]
                    )
                    + "\n"
                )
                cov_dict[context] += cov
                mc_dict[context] += unconverted_c
                out += data
                cur_out_pos += len(data)

        elif fields[2] == "G":
            pos = int(fields[1]) - 1
            try:
                context = "".join(
                    [
                        complement[base]
                        for base in reversed(seq[(pos - num_downstr_bases) : (pos + num_upstr_bases + 1)])
                    ]
                )
            except Exception:  # complete context is not available, skip
                continue
            unconverted_c = fields[4].count(",")
            converted_c = fields[4].count("a")
            cov = unconverted_c + converted_c
            if cov > 0 and len(context) == context_len:
                line_counts += 1
                data = (
                    "\t".join(
                        [
                            cur_chrom,
                            str(pos + 1),  # ALLC pos is 1-based
                            "-",
                            context,
                            str(unconverted_c),
                            str(cov),
                            "1",
                        ]
                    )
                    + "\n"
                )
                cov_dict[context] += cov
                mc_dict[context] += unconverted_c
                out += data
                cur_out_pos += len(data)

        if line_counts > buffer_line_number:
            output_file_handler.write(out)
            line_counts = 0
            out = ""

    if line_counts > 0:
        output_file_handler.write(out)
    result_handle.close()
    output_file_handler.close()

    if tabix:
        subprocess.run(shlex.split(f"tabix -b 2 -e 2 -s 1 {output_path}"), check=True)

    count_df = pd.DataFrame({"mc": mc_dict, "cov": cov_dict})
    count_df["mc_rate"] = count_df["mc"] / count_df["cov"]

    total_genome_length = fai_df["LENGTH"].sum()
    count_df["genome_cov"] = total_line / total_genome_length

    if save_count_df:
        count_df.to_csv(output_path + ".count.csv")
        return None
    else:
        return count_df


def _aggregate_count_df(count_dfs):
    total_df = pd.concat(count_dfs)
    total_df = total_df.groupby(total_df.index).sum()
    total_df["mc_rate"] = total_df["mc"] / total_df["cov"]
    total_df["mc"] = total_df["mc"].astype(int)
    total_df["cov"] = total_df["cov"].astype(int)
    return total_df


@doc_params(
    compress_level_doc=compress_level_doc,
    cpu_basic_doc=cpu_basic_doc,
    reference_fasta_doc=reference_fasta_doc,
    convert_bam_strandness_doc=convert_bam_strandness_doc,
)
def bam_to_allc(
    bam_path,
    reference_fasta,
    output_path=None,
    cpu=1,
    num_upstr_bases=0,
    num_downstr_bases=2,
    min_mapq=10,
    min_base_quality=20,
    compress_level=5,
    save_count_df=False,
    convert_bam_strandness=False,
):
    """\
    Generate 1 ALLC file from 1 position sorted BAM file via samtools mpileup.

    Parameters
    ----------
    bam_path
        Path to 1 position sorted BAM file
    reference_fasta
        {reference_fasta_doc}
    output_path
        Path to 1 output ALLC file
    cpu
        {cpu_basic_doc} DO NOT use cpu > 1 for single cell ALLC generation.
        Parallel on cell level is better for single cell project.
    num_upstr_bases
        Number of upstream base(s) of the C base to include in ALLC context column,
        usually use 0 for normal BS-seq, 1 for NOMe-seq.
    num_downstr_bases
        Number of downstream base(s) of the C base to include in ALLC context column,
        usually use 2 for both BS-seq and NOMe-seq.
    min_mapq
        Minimum MAPQ for a read being considered, samtools mpileup parameter, see samtools documentation.
    min_base_quality
        Minimum base quality for a base being considered, samtools mpileup parameter,
        see samtools documentation.
    compress_level
        {compress_level_doc}
    save_count_df
        If true, save an ALLC context count table next to ALLC file.
    convert_bam_strandness
        {convert_bam_strandness_doc}

    Returns
    -------
    count_df
        a pandas.DataFrame for overall mC and cov count separated by mC context.
    """
    buffer_line_number = 100000
    tabix = True

    # Check fasta index
    if not pathlib.Path(reference_fasta).exists():
        raise FileNotFoundError(f"Reference fasta not found at {reference_fasta}.")
    if not pathlib.Path(reference_fasta + ".fai").exists():
        raise FileNotFoundError("Reference fasta not indexed. Use samtools faidx to index it and run again.")
    fai_df = _read_faidx(pathlib.Path(reference_fasta + ".fai"))

    if convert_bam_strandness:
        temp_bam_path = f"{output_path}.temp.bam"
        _convert_bam_strandness(bam_path, temp_bam_path)
        bam_path = temp_bam_path

    if not pathlib.Path(bam_path + ".bai").exists():
        subprocess.check_call(shlex.split("samtools index " + bam_path))

    # check chromosome between BAM and FASTA
    # samtools have a bug when chromosome not match...
    bam_chroms_index = _get_bam_chrom_index(bam_path)
    unknown_chroms = [i for i in bam_chroms_index if i not in fai_df.index]
    if len(unknown_chroms) != 0:
        unknown_chroms = " ".join(unknown_chroms)
        raise IndexError(
            f"BAM file contain unknown chromosomes: {unknown_chroms}\n"
            "Make sure you use the same genome FASTA file for mapping and bam-to-allc."
        )

    # if parallel, chunk genome
    if cpu > 1:
        regions = genome_region_chunks(reference_fasta + ".fai", bin_length=100000000, combine_small=False)
    else:
        regions = None

    # Output path
    input_path = pathlib.Path(bam_path)
    file_dir = input_path.parent
    if output_path is None:
        allc_name = "allc_" + input_path.name.split(".")[0] + ".tsv.gz"
        output_path = str(file_dir / allc_name)
    else:
        if not output_path.endswith(".gz"):
            output_path += ".gz"

    if cpu > 1:
        raise NotImplementedError

        temp_out_paths = []
        for batch_id in range(len(regions)):
            temp_out_paths.append(output_path + f".batch_{batch_id}.tmp.tsv.gz")

        with ProcessPoolExecutor(max_workers=cpu) as executor:
            future_dict = {}
            for batch_id, (region, out_temp_path) in enumerate(zip(regions, temp_out_paths)):
                _kwargs = {
                    "bam_path": bam_path,
                    "reference_fasta": reference_fasta,
                    "fai_df": fai_df,
                    "output_path": out_temp_path,
                    "region": region,
                    "num_upstr_bases": num_upstr_bases,
                    "num_downstr_bases": num_downstr_bases,
                    "buffer_line_number": buffer_line_number,
                    "min_mapq": min_mapq,
                    "min_base_quality": min_base_quality,
                    "compress_level": compress_level,
                    "tabix": False,
                    "save_count_df": False,
                }
                future_dict[executor.submit(_bam_to_allc_worker, **_kwargs)] = batch_id

            count_dfs = []
            for future in as_completed(future_dict):
                batch_id = future_dict[future]
                try:
                    count_dfs.append(future.result())
                except Exception as exc:
                    log.info(f"{batch_id!r} generated an exception: {exc}")

            # aggregate ALLC
            with open_allc(
                output_path,
                mode="w",
                compresslevel=compress_level,
                threads=1,
                region=None,
            ) as out_f:
                # TODO: Parallel ALLC is overlapped,
                #  the split by region strategy only split reads, but reads can overlap
                # need to adjust and merge end rows in aggregate ALLC
                for out_temp_path in temp_out_paths:
                    with open_allc(out_temp_path) as f:
                        for line in f:
                            out_f.write(line)
                    subprocess.check_call(["rm", "-f", out_temp_path])

            # tabix
            if tabix:
                subprocess.run(shlex.split(f"tabix -b 2 -e 2 -s 1 {output_path}"), check=True)

            # aggregate count_df
            count_df = _aggregate_count_df(count_dfs)
            if save_count_df:
                count_df.to_csv(output_path + ".count.csv")

            # clean up temp bam
            if convert_bam_strandness:
                # this bam path is the temp file path
                subprocess.check_call(["rm", "-f", bam_path])
                subprocess.check_call(["rm", "-f", f"{bam_path}.bai"])
            return count_df
    else:
        result = _bam_to_allc_worker(
            bam_path,
            reference_fasta,
            fai_df,
            output_path,
            region=None,
            num_upstr_bases=num_upstr_bases,
            num_downstr_bases=num_downstr_bases,
            buffer_line_number=buffer_line_number,
            min_mapq=min_mapq,
            min_base_quality=min_base_quality,
            compress_level=compress_level,
            tabix=tabix,
            save_count_df=save_count_df,
        )

        # clean up temp bam
        if convert_bam_strandness:
            # this bam path is the temp file path
            subprocess.check_call(["rm", "-f", bam_path])
            subprocess.check_call(["rm", "-f", f"{bam_path}.bai"])

        return result
