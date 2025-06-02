
import argparse
import pandas as pd
from Bio import SeqIO
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_gtf(gtf_file):
    """Extract exons from GTF and group them by transcript ID."""
    transcript_exons = {}

    with open(gtf_file, "r") as f:
        for line in f:
            if line.startswith("#"):  # Ignore comments
                continue
            
            cols = line.strip().split("\t")
            if cols[2] == "exon":  # Only consider exons
                chrom = cols[0]
                start, end = int(cols[3]), int(cols[4])
                strand = cols[6]
                transcript_id = cols[-1].split('transcript_id "')[1].split('"')[0]
                
                if transcript_id not in transcript_exons:
                    transcript_exons[transcript_id] = []
                
                transcript_exons[transcript_id].append((chrom, start, end, strand))
    
    return transcript_exons


def extract_splice_junctions(transcript_exons):
    """Identify splice junctions (introns) from exon positions."""
    splice_junctions = []

    for transcript, exons in transcript_exons.items():
        exons.sort(key=lambda x: x[1])  # Sort exons by start position

        for i in range(len(exons) - 1):
            chrom, exon_end, strand = exons[i][0], exons[i][2], exons[i][3]  # End of exon i
            next_start = exons[i + 1][1]  # Start of exon i+1
            
            # Introns are between consecutive exons
            splice_junctions.append([chrom, exon_end, next_start, f"JUNC_{len(splice_junctions):08d}", 1, strand, transcript])

    return splice_junctions


def load_fasta(fasta_file):
    """Load full chromosome sequences from a genome FASTA file."""
    genome = {}
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            genome[record.id] = str(record.seq).upper()
    return genome


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return "".join(complement.get(base, "N") for base in reversed(seq))


def extract_splice_sequences(junctions, genome, flank_short, flank_long):
    """Extract 200 bp donor and acceptor site sequences considering strand orientation."""
    expected_length = flank_short + flank_long + 2
    sequences = []

    for chrom, start, end, name, score, strand, transcripts in junctions:
        end = end - 1  # Convert to 0-based end coordinate
        if chrom not in genome:
            continue  # Skip missing chromosomes

        if strand == "+":  # Positive strand (normal direction)
            seq = genome[chrom]
            # Donor: 3bp upstream, then 'GT', then 30bp downstream
            donor_left  = seq[max(0, start - flank_short): start]
            donor_motif = seq[start: start+2]
            donor_right = seq[start+2: start+2+flank_long]
            donor_seq = donor_left + donor_motif + donor_right

            # Acceptor: 30bp upstream, then 'AG', then 3bp downstream
            acceptor_left  = seq[max(0, end-2-flank_long): end-2]
            acceptor_motif = seq[end-2: end]
            acceptor_right = seq[end: end+flank_short]
            acceptor_seq = acceptor_left + acceptor_motif + acceptor_right

        else:  # Negative strand (reverse complement)
            seq = genome[chrom]
            # For negative strand, the donor site is at the end coordinate.
            # Extract from (end - 2 - flank_long) to (end + flank_short) and reverse complement.
            donor_raw = seq[max(0, end - 2 - flank_long): end + flank_short]
            donor_seq = reverse_complement(donor_raw)

            # For negative strand, the acceptor site is at the start coordinate.
            # Extract from (start - flank_short) to (start + 2 + flank_long) and reverse complement.
            acceptor_raw = seq[max(0, start - flank_short): start + 2 + flank_long]
            acceptor_seq = reverse_complement(acceptor_raw)

        # Ensure correct length
        if len(donor_seq) == expected_length and len(acceptor_seq) == expected_length:
            sequences.append((name, donor_seq, acceptor_seq, strand, transcripts))

    return sequences


class SpliceSiteCNN(nn.Module):
    def __init__(self, seq_length):
        super(SpliceSiteCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Temporary dummy input to determine flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, seq_length)
            out = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))
            self.flattened_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x.squeeze()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def load_model_cnn(model_path, seq_length):
    model = SpliceSiteCNN(seq_length=seq_length)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))  # Load saved weights
    model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model


def one_hot_encode(seq, seq_length):
    """One-hot encode a DNA sequence to a shape (4, seq_length)."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded = np.zeros((4, seq_length), dtype=np.float32)

    for i, base in enumerate(seq[:seq_length]):  # Trim to fixed length
        encoded[:, i] = mapping.get(base, [0, 0, 0, 0])  # Handle invalid bases (N, etc.)

    return encoded


def predict_splice_sites(model, sequences, batch_size=128):
    """Predict the probability of each sequence being a true splice site using batch processing."""
    device = next(model.parameters()).device  # Ensure model and data are on the same device
    model.eval()
    junction_scores = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(np.array(sequences[i:i+batch_size])).float().to(device)
            outputs = model(batch)  # Forward pass
            junction_scores.append(outputs.cpu())  # Move only the batch to CPU
    return torch.cat(junction_scores).numpy().tolist()  # Convert to a list once


def save_splice_scores(splice_sequences, junction_scores, output_file):
    """
    Save CNN-predicted splice site scores to a TSV file.
    """
    results = []
    for i, (_, _, _, strand, transcripts) in enumerate(splice_sequences):
        results.append([transcripts, strand, junction_scores[i]])
    df = pd.DataFrame(results, columns=[ "Transcripts", "Strand", "Junction_Score"])
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Splice site scores saved to {output_file}")
    
    make_plot(df, output_file.replace(".tsv", "_distribution_plot.png"))
    return df


def make_plot(df, output_file):
    junction_scores = df["Junction_Score"]

    # Zoom in on the central 1–99th percentile (adjust or set fixed window if you prefer)
    lower = junction_scores.quantile(0.01)
    upper = junction_scores.quantile(0.99)

    plt.figure(figsize=(8, 5))
    plt.hist(
        junction_scores,
        bins=100,
        range=(lower, upper),
        edgecolor='black',
        alpha=0.7,
        color='orange'
    )
    plt.xlim(lower, upper)
    plt.xlabel("Junction Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of junction scores for each transcript for drosophila")
    plt.grid(True)
    plt.tight_layout()

    # Save to current directory
    plt.savefig(output_file, dpi=300)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Train splicing model with input junctions and reference genome")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model")
    parser.add_argument("--reference_genome", type=str, required=True,
                        help="Path to reference genome FASTA file")
    parser.add_argument("--stringtie_gtf", type=str, required=True,
                        help="Path to stringtie gtf file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file name (species-specific)")

    args = parser.parse_args()

    model_path = args.model_path
    fasta_file = args.reference_genome
    stringtie_gtf = args.stringtie_gtf
    output_file = args.output_file

    flanking_short = 3
    flanking_long = 35
    seq_len = (flanking_short + flanking_long + 2) * 2

    stringtie_introns = parse_gtf(stringtie_gtf)
    splice_junctions = extract_splice_junctions(stringtie_introns)
    genome = load_fasta(fasta_file)
    splice_sequences = extract_splice_sequences(splice_junctions, genome, flanking_short, flanking_long)
    print(f"Extracted {len(splice_sequences)} donor/acceptor site sequences.")

    trained_model = load_model_cnn(model_path, seq_len)
    encoded_sequences = [one_hot_encode(seq[1] + seq[2],seq_length=seq_len) for seq in splice_sequences]
    junction_scores = predict_splice_sites(trained_model, encoded_sequences)

    df = save_splice_scores(splice_sequences, junction_scores, output_file)
    df_min_scores = df.groupby("Transcripts", as_index=False).agg({
        "Strand": "first",
        "Junction_Score": "min"
    })
    
    df_min_scores.to_csv("min_" + output_file, sep="\t", index=False)
    print(f"Minimum junction scores saved to min_{output_file}")

    make_plot(df_min_scores, output_file.replace(".tsv", "_min_distribution_plot.png"))

    elapsed_time = time.time() - start_time
    print(f"Total run time: {elapsed_time:.2f} seconds")

    
if __name__ == "__main__":
    main()
