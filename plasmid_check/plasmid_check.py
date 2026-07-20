def check_sanger_sequence(
    dna_seq_name,
    dna_seq,
    sanger_seq_name,
    sanger_seq,
    plot=False,
    plot_outfile=None,
    plot_width=10,
    gap_penalty=-10,
    target_start=None,
    target_end=None,
):
    """Compare whole plasmid sequencing reads to reference template."""

    template_len = len(dna_seq)
    sample_len = len(sanger_seq)

    # Print length information
    print(f"-> Template ({dna_seq_name}) Length: {template_len} bp")
    print(f"-> Sample ({sanger_seq_name}) Length: {sample_len} bp")

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
                "sanger_seq_name",
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

    ali = pairwise_dna(dna_seq, sanger_seq, gap_penalty=gap_penalty)

    sanger_dna_start = ali.trace[0][0]
    sanger_dna_end = ali.trace[-1][0]

    mismatches = compare_sequences(
        dna_seq,
        sanger_seq,
        ali,
        target_start=target_start,
        target_end=target_end,
    )

    print(f"{dna_seq_name}, {sanger_seq_name}:")
    print(f"\tdna_seq: {dna_seq}")
    print(f"\tsanger_seq: {sanger_seq}")
    print(
        f"\tAlignment Nucleotides: {sanger_dna_start+1} to {sanger_dna_end+1}"
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
            alphabet_1, alphabet_2, score_matrix
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

    df_rows = []
    for m in mismatches:
        df_rows.append(
            [
                dna_seq_name,
                sanger_seq_name,
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
            "sanger_seq_name",
            "region",
            "dna_pos",
            "ref_seq",
            "alt_seq",
            "aa_pos",
            "ref_aa",
            "alt_aa",
        ],
    )
