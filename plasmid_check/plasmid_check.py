from dataclasses import dataclass
from Bio import SeqIO
from Bio.Seq import Seq
from biotite import sequence as bioseq
import biotite.sequence.align as bioalign
import biotite.sequence.graphics as graphics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class MisMatch:
    """Class for mismatches between two sequences."""

    seq1_pos: int  # 0-based nucleotide position
    ref_nt: str  # Reference nucleotide/codon
    alt_nt: str  # Sample nucleotide/codon
    region: str  # 'Target Gene' or 'Backbone'
    aa_pos: int = None  # 1-based AA residue relative to gene start
    ref_aa: str = None  # Reference amino acid
    alt_aa: str = None  # Sample amino acid


def parse_fasta(fasta_file):
    """Read a fasta file and return a pandas dataframe."""
    records = SeqIO.parse(fasta_file, "fasta")
    df = pd.DataFrame(columns=["seq"])
    for record in records:
        df.loc[record.id] = [str(record.seq)]
    return df


def pairwise_dna(seq1, seq2, type="local", gap_penalty=-10):
    """Pairwise alignment of two DNA sequences."""
    seq1 = bioseq.NucleotideSequence(seq1)
    seq2 = bioseq.NucleotideSequence(seq2)
    matrix = bioalign.SubstitutionMatrix.std_nucleotide_matrix()
    local = True if type == "local" else False
    ali = bioalign.align_optimal(
        seq1,
        seq2,
        matrix,
        local=local,
        gap_penalty=gap_penalty,
    )[0]
    return ali


def auto_assign(sanger_df, dna_df, gap_penalty=-10):
    """Auto assign sanger sequences to reference sequences.

    Args:
        sanger_df (pd.DataFrame): Dataframe of sanger sequences.
        dna_df (pd.DataFrame): Dataframe of reference sequences.

        DataFrames must have a column named "seq" with the sequence and index of
            the DataFrame must be the sequence name.

    Returns:
        data_df: Dataframe with sanger sequences assigned to reference sequences.
    """
    # Auto assign sanger sequences to reference sequences

    score_df = pd.DataFrame(columns=["sanger_seq", "dna_seq", "score", "orientation"])

    # Compute pairwise scores for all sanger sequences against all dna sequences

    for sanger_seq_name in (pbar := tqdm(sanger_df.index)):
        sanger_seq = sanger_df.loc[sanger_seq_name, "seq"]
        rev_sanger_seq = str(Seq(sanger_seq).reverse_complement())
        for dna_seq_name in dna_df.index:
            dna_seq = dna_df.loc[dna_seq_name, "seq"]

            pbar.set_description(f"Checking {sanger_seq_name} against {dna_seq_name}")
            alignment = pairwise_dna(sanger_seq, dna_seq, gap_penalty=gap_penalty)
            rev_alignment = pairwise_dna(
                rev_sanger_seq, dna_seq, gap_penalty=gap_penalty
            )
            score = max(alignment.score, rev_alignment.score)
            orientation = (
                "forward" if alignment.score > rev_alignment.score else "reverse"
            )

            score_df.loc[len(score_df)] = [
                sanger_seq_name,
                dna_seq_name,
                score,
                orientation,
            ]

    # Assign sanger sequences to dna sequences
    data_df = pd.DataFrame(columns=["dna_seq", "orientation"])
    for sanger_seq_name in score_df["sanger_seq"].unique():
        # Get best dna sequence
        best_dna_seq = (
            score_df[score_df["sanger_seq"] == sanger_seq_name]
            .sort_values("score", ascending=False)
            .iloc[0]["dna_seq"]
        )

        # Get orientation
        orientation = (
            score_df[score_df["sanger_seq"] == sanger_seq_name]
            .sort_values("score", ascending=False)
            .iloc[0]["orientation"]
        )

        # Add to data dict
        data_df.loc[sanger_seq_name] = [best_dna_seq, orientation]

    return data_df


def compare_sequences(
    seq1, seq2, alignment, target_start=None, target_end=None
):
    """Compare two sequences nucleotide-by-nucleotide and codon-by-codon."""
    mismatches = []
    seq1_ali_end = alignment.trace[-1][0]

    # Map trace to numpy arrays for direct lookups
    trace_seq1 = alignment.trace[:, 0]
    trace_seq2 = alignment.trace[:, 1]

    for idx, (seq1_i, seq2_i) in enumerate(alignment.trace):
        if seq1_i == -1 or seq2_i == -1:
            continue  # Skip unaligned terminal regions or insertions/deletions for simple SNP checks

        ref_nt = seq1[seq1_i]
        alt_nt = seq2[seq2_i]

        if ref_nt != alt_nt:
            pos_1based = seq1_i + 1

            # Check if nucleotide falls within Target Gene boundaries
            is_target = False
            if target_start and target_end and target_start > 0:
                if target_start <= target_end:
                    is_target = target_start <= pos_1based <= target_end
                else:
                    is_target = (
                        pos_1based >= target_start or pos_1based <= target_end
                    )

            if is_target:
                # Target Gene Logic: Codon & Amino Acid position relative to target_start
                gene_nt_offset = pos_1based - target_start
                codon_start_pos = (
                    target_start - 1 + (gene_nt_offset // 3) * 3
                )
                aa_residue_num = (gene_nt_offset // 3) + 1

                # Extract codon triplet if available within alignment
                if codon_start_pos + 3 <= len(seq1):
                    ref_codon = seq1[codon_start_pos : codon_start_pos + 3]

                    # Map corresponding sample nucleotides for this triplet
                    alt_codon_chars = []
                    for p in range(
                        codon_start_pos, codon_start_pos + 3
                    ):
                        matches = np.where(trace_seq1 == p)[0]
                        if len(matches) > 0 and trace_seq2[matches[0]] != -1:
                            alt_codon_chars.append(seq2[trace_seq2[matches[0]]])
                        else:
                            alt_codon_chars.append("-")
                    alt_codon = "".join(alt_codon_chars)

                    ref_aa = (
                        str(Seq(ref_codon).translate())
                        if "-" not in ref_codon
                        else "-"
                    )
                    alt_aa = (
                        str(Seq(alt_codon).translate())
                        if "-" not in alt_codon
                        else "-"
                    )

                    # Deduplicate if codon mismatch was already processed
                    if not any(
                        m.seq1_pos == codon_start_pos
                        and m.region == "Target Gene"
                        for m in mismatches
                    ):
                        mismatches.append(
                            MisMatch(
                                seq1_pos=codon_start_pos,
                                ref_nt=ref_codon,
                                alt_nt=alt_codon,
                                region="Target Gene",
                                aa_pos=aa_residue_num,
                                ref_aa=ref_aa,
                                alt_aa=alt_aa,
                            )
                        )
            else:
                # Backbone Logic: Individual Nucleotide Mismatches
                mismatches.append(
                    MisMatch(
                        seq1_pos=seq1_i,
                        ref_nt=ref_nt,
                        alt_nt=alt_nt,
                        region="Backbone",
                    )
                )

    return mismatches


def check_plasmid_sequence(
    dna_seq_name,
    dna_seq,
    read_seq_name,
    read_seq,
    gap_penalty=-10,
    target_start=None,
    target_end=None,
):
    """Compare whole plasmid sequencing reads to reference template."""

    template_len = len(dna_seq)
    sample_len = len(read_seq)

    # Print length information
    print(f"-> Template ({dna_seq_name}) Length: {template_len} bp")
    print(f"-> Sample ({read_seq_name}) Length: {sample_len} bp")

    # --- EARLY EXIT IF LENGTHS DO NOT MATCH ---
    if template_len != sample_len:
        print(
            f"-> Result: Length mismatch! Difference is {abs(template_len - sample_len)} bp"
        )
        print("   Skipping alignment due to length mismatch.\n")

        # Return empty DataFrame with expected layout
        return pd.DataFrame(
            columns=[
                "dna_seq_name",
                "read_seq_name",
                "region",
                "dna_pos",
                "ref_seq",
                "alt_seq",
                "aa_pos",
                "ref_aa",
                "alt_aa",
            ]
        )

    # --- IF LENGTHS MATCH, RUN ALIGNMENT ---
    print("-> Result: Lengths match perfectly!")

    ali = pairwise_dna(dna_seq, read_seq, gap_penalty=gap_penalty)

    read_dna_start = ali.trace[0][0]
    read_dna_end = ali.trace[-1][0]

    mismatches = compare_sequences(
        dna_seq,
        read_seq,
        ali,
        target_start=target_start,
        target_end=target_end,
    )

    print(f"{dna_seq_name}, {read_seq_name}:")
    print(f"\tdna_seq: {dna_seq}")
    print(f"\tread_seq: {read_seq}")
    print(
        f"\tAlignment Nucleotides: {read_dna_start+1} to {read_dna_end+1}"
    )
    print(f"\tMismatches:")

    target_mismatches = [m for m in mismatches if m.region == "Target Gene"]
    backbone_mismatches = [m for m in mismatches if m.region == "Backbone"]

    print("Target gene:")
    if target_mismatches:
        for m in target_mismatches:
            start_nt = m.seq1_pos + 1
            end_nt = m.seq1_pos + 3
            print(
                f"\tNucleotides {start_nt}-{end_nt} {m.ref_nt} ({m.ref_aa}{m.aa_pos}) -> {m.alt_nt} ({m.alt_aa})"
            )
    else:
        print("\tNone")

    print("Backbone:")
    if backbone_mismatches:
        for m in backbone_mismatches:
            pos_nt = m.seq1_pos + 1
            print(f"\tNucleotides: {pos_nt} {m.ref_nt} -> {m.alt_nt}")
    else:
        print("\tNone")

    df_rows = []
    for m in mismatches:
        df_rows.append(
            [
                dna_seq_name,
                read_seq_name,
                m.region,
                m.seq1_pos + 1,
                m.ref_nt,
                m.alt_nt,
                m.aa_pos if m.aa_pos else "-",
                m.ref_aa if m.ref_aa else "-",
                m.alt_aa if m.alt_aa else "-",
            ]
        )

    return pd.DataFrame(
        df_rows,
        columns=[
            "dna_seq_name",
            "read_seq_name",
            "region",
            "dna_pos",
            "ref_seq",
            "alt_seq",
            "aa_pos",
            "ref_aa",
            "alt_aa",
        ],
    )