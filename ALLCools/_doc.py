from textwrap import dedent

compress_level_doc = "Compression level for the output file"

idx_doc = "If true, save an methylpy chromosome index for back compatibility. " \
          "If you only use methylpy to call DMR, this don't need to be True."

allc_path_doc = "Path to 1 ALLC file"

allc_paths_doc = "Single ALLC path contain wildcard OR multiple space separated ALLC paths " \
                 "OR a file contains 1 ALLC path in each row."

cpu_basic_doc = 'Number of processes to use in parallel.'

chrom_size_path_doc = "Path to UCSC chrom size file. " \
                      "This can be generated from the genome fasta or downloaded via UCSC fetchChromSizes tools. " \
                      "All ALLCools functions will refer to this file whenever possible to check for " \
                      "chromosome names and lengths, so it is crucial to use a chrom size file consistent " \
                      "to the reference fasta file ever since mapping. " \
                      "ALLCools functions will not change or infer chromosome names."

remove_additional_chrom_doc = "Whether to remove rows with unknown chromosome instead of raising KeyError"

mc_contexts_doc = "Space separated mC context patterns to extract from ALLC. " \
                  "The context length should be the same as ALLC file context. " \
                  "Context pattern follows IUPAC nucleotide code, e.g. N for ATCG, H for ATC, Y for CT."

cov_cutoff_doc = "Max cov filter for a single site in ALLC. Sites with cov > cov_cutoff will be skipped."


def doc_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__doc__ = dedent(obj.__doc__).format(**kwds)
        return obj

    return dec