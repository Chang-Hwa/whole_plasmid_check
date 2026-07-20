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

    seq1_pos: int
    seq1_codon: str
    seq1_aa: str
    seq1_resi: int
    seq2_pos: int
    seq2_codon: str
    seq2_aa: str
    seq2_resi: int
    # HIGHLIGHT: Added region field to data structure
    region: str = "Plasmid Backbone"


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
    """Auto assign sanger sequences to reference sequences."""

    score_df = pd.DataFrame(
        columns=["sanger_seq", "dna_seq", "score", "orientation"]
    )

    for sanger_seq_name in (pbar := tqdm(sanger_df.index)):
        sanger_seq = sanger_df.loc[sanger_seq_name, "seq"]
        rev_sanger_seq = str(Seq(sanger_seq).reverse_complement())
        for dna_seq_name in dna_df.index:
            dna_seq = dna_df.loc[dna_seq_name, "seq"]

            pbar.set_description(
                f"Checking {sanger_seq_name} against {dna_seq_name}"
            )
            alignment = pairwise_dna(
                sanger_seq, dna_seq, gap_penalty=gap_penalty
            )
            rev_alignment = pairwise_dna(
                rev_sanger_seq, dna_seq, gap_penalty=gap_penalty
            )
            score = max(alignment.score, rev_alignment.score)
            orientation = (
                "forward"
                if alignment.score > rev_alignment.score
                else "reverse"
            )

            score_df.loc[len(score_df)] = [
                sanger_seq_name,
                dna_seq_name,
                score,
                orientation,
            ]

    data_df = pd.DataFrame(columns=["dna_seq", "orientation"])
    for sanger_seq_name in score_df["sanger_seq"].unique():
        best_dna_seq = (
            score_df[score_df["sanger_seq"] == sanger_seq_name]
            .sort_values("score", ascending=False)
            .iloc[0]["dna_seq"]
        )

        orientation = (
            score_df[score_df["sanger_seq"] == sanger_seq_name]
            .sort_values("score", ascending=False)
            .iloc[0]["orientation"]
        )

        data_df.loc[sanger_seq_name] = [best_dna_seq, orientation]

    return data_df


# HIGHLIGHT: Added target_start and target_end parameter defaults
def compare_sequences(
    seq1, seq2, alignment, target_start=None, target_end=None
):
    """Compare two sequences and return mismatches."""

    mismatches = []

    seq1_ali_start = alignment.trace[0][0]
    seq1_ali_end = alignment.trace[-1][0]

    for seq1_i, seq2_i in alignment.trace:
        if seq1_i % 3 == 0:
            if seq1_i + 3 > seq1_ali_end:
                break

            seq1_index = np.where(alignment.trace[:, 0] == seq1_i)[0][0]
            seq1_j = alignment.trace[seq1_index + 1][0]
            seq1_k = alignment.trace[seq1_index + 2][0]
            seq2_j = alignment.trace[seq1_index + 1][1]
            seq2_k = alignment.trace[seq1_index + 2][1]
            seq1_indicies = [alignment.trace[seq1_index][0], seq1_j, seq1_k]
            seq2_indicies = [alignment.trace[seq1_index][1], seq2_j, seq2_k]
            seq1_codon = "".join(
                [seq1[i] if i != -1 else "-" for i in seq1_indicies]
            )
            seq2_codon = "".join(
                [seq2[i] if i != -1 else "-" for i in seq2_indicies]
            )

            seq2_codon = "".join(
                [
                    seq2_codon[i] if seq2_codon[i] != "N" else seq1_codon[i]
                    for i in range(3)
                ]
            )

            if seq1_codon != seq2_codon:
                seq1_aa = None
                seq2_aa = None
                if ("-" not in seq1_codon) and ("N" not in seq1_codon):
                    seq1_aa = str(Seq(seq1_codon).translate())
                if ("-" not in seq2_codon) and ("N" not in seq2_codon):
                    seq2_aa = str(Seq(seq2_codon).translate())

                # HIGHLIGHT: Check region assignment (1-based position conversion)
                pos_1based = seq1_i + 1
                region = "Plasmid Backbone"
                if (
                    target_start is not None
                    and target_end is not None
                    and target_start > 0
                    and target_end > 0
                ):
                    if target_start <= target_end:
                        if target_start <= pos_1based <= target_end:
                            region = "Target Gene"
                    else:
                        # Handle origin wrap arounds
                        if (
                            pos_1based >= target_start
                            or pos_1based <= target_end
                        ):
                            region = "Target Gene"

                mismatch = MisMatch(
                    seq1_pos=seq1_i,
                    seq1_codon=seq1_codon,
                    seq1_aa=seq1_aa,
                    seq1_resi=seq1_i // 3 + 1,
                    seq2_pos=seq2_i,
                    seq2_codon=seq2_codon,
                    seq2_aa=seq2_aa,
                    seq2_resi=seq2_i // 3 + 1,
                    region=region,  # HIGHLIGHT: Assigned region to object
                )

                mismatches.append(mismatch)
    return mismatches


# HIGHLIGHT: Added target_start and target_end inputs to check_sanger_sequence
def check_sanger_sequence(
    dna_seq_name,
    dna_seq,
    sanger_seq_name,
    sanger_seq,
    plot=False,
    plot_outfile=None,
    plot_width=10,
    gap_penalty=-10,
    target_start=None,  # HIGHLIGHT: New parameter
    target_end=None,  # HIGHLIGHT: New parameter
):
    """Compare sanger sequences to reference sequence."""
    ali = pairwise_dna(dna_seq, sanger_seq, gap_penalty=gap_penalty)

    sanger_dna_start = ali.trace[0][0]
    sanger_dna_end = ali.trace[-1][0]

    print(f"{dna_seq_name}, {sanger_seq_name}:")
    print(f"\tdna_seq: {dna_seq}")
    print(f"\tsanger_seq: {sanger_seq}")
    print(f"\tAlignment Nucleotides: {sanger_dna_start+1} to {sanger_dna_end+1}")

    # HIGHLIGHT: Passed target boundary parameters into compare_sequences
    mismatches = compare_sequences(
        dna_seq,
        sanger_seq,
        ali,
        target_start=target_start,
        target_end=target_end,
    )

    if mismatches:
        print("\tMismatches:")
        for mismatch in mismatches:
            print(
                f"\t\t[{mismatch.region}] Nucleotides {mismatch.seq1_pos+1}-{mismatch.seq1_pos+4} "
                f"{mismatch.seq1_codon} ({mismatch.seq1_aa}{mismatch.seq1_resi}) -> "
                f"{mismatch.seq2_codon} ({mismatch.seq2_aa})"
            )
    else:
        print("\tNo changes found in sanger sequence")

    if plot:
        aln_len = len(ali.trace)
        plot_height = max(int(plot_width * aln_len / 400), plot_width)

        fig, axs = plt.subplots(
            nrows=1, ncols=1, figsize=(plot_height, plot_width)
        )
        plt.suptitle(f"{dna_seq_name}\n{sanger_seq_name}")

        std_matrix = bioalign.SubstitutionMatrix.std_nucleotide_matrix()
        alphabet_1 = std_matrix.get_alphabet1()
        alphabet_2 = std_matrix.get_alphabet2()
        score_matrix = np.ones((len(alphabet_1), len(alphabet_2)))

        for i in range(4):
            for j in range(4):
                score_matrix[i, j] = 10

        for i in range(len(alphabet_1)):
            score_matrix[i, i] = 0

        matrix = bioalign.SubstitutionMatrix(
            alphabet_1,
            alphabet_2,
            score_matrix,
        )

        graphics.plot_alignment_similarity_based(
            axs,
            ali,
            matrix=matrix,
            labels=["DNA Sequence", "Sequencing Read"],
            show_numbers=True,
            show_line_position=True,
        )

        plt.tight_layout()

        if plot_outfile:
            plt.savefig(plot_outfile, dpi=300)

        plt.show()
        plt.close()

    # HIGHLIGHT: Added "region" column to the returned DataFrame
    return pd.DataFrame(
        [
            [
                dna_seq_name,
                sanger_seq_name,
                mismatch.region,  # HIGHLIGHT: Region output added
                mismatch.seq1_pos,
                mismatch.seq1_codon,
                mismatch.seq1_aa,
                mismatch.seq1_resi,
                mismatch.seq2_pos,
                mismatch.seq2_codon,
                mismatch.seq2_aa,
                mismatch.seq2_resi,
                "".join(
                    [dna_seq[x] if x != -1 else "-" for x in ali.trace[:, 0]]
                ),
                "".join(
                    [sanger_seq[x] if x != -1 else "-" for x in ali.trace[:, 1]]
                ),
            ]
            for mismatch in mismatches
        ],
        columns=[
            "dna_seq_name",
            "sanger_seq_name",
            "region",  # HIGHLIGHT: Region column header added
            "dna_pos",
            "dna_codon",
            "dna_aa",
            "dna_resi",
            "sanger_pos",
            "sanger_codon",
            "sanger_aa",
            "sanger_resi",
            "dna_seq_aln",
            "sanger_seq_aln",
        ],
    )
