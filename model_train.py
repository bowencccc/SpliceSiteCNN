
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


def generate_positive_introns(reliable_bed):
    # Read the reliable BED file
    positives_introns = pd.read_csv(reliable_bed, sep="\t", names=["Chromosome", "Start", "End", "Junction_ID", "Score", "Strand"], comment="t")
    return positives_introns


def generate_more_difficult_negatives_introns(tiecov_bed):
    # Read the TieCov BED file
    df = pd.read_csv(tiecov_bed, sep="\t", names=["Chromosome", "Start", "End", "Junction_ID", "Score", "Strand"], comment="t")
    # Filter out junctions with coverage equal to 3 or less
    coverage_less_than_or_equal_to_3 = df[df["Score"] <= 3]
    return coverage_less_than_or_equal_to_3


def extract_sequences_from_introns(introns, genome_fasta,flank_short = 3, flank_long = 35):
    sequences = []
    expected_length = flank_short + 2 + flank_long
    # Extract sequences from the genome FASTA file
    genome_dict = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))
    genome_str  = {
        chrom: str(rec.seq).upper()
        for chrom, rec in genome_dict.items()
    }
    genome_dict.clear()
    # Find GT-AG sequences corresponding to the introns
    for row in introns.itertuples(index=True, name="Junction"):
        start = row.Start
        end = row.End
        chrom = row.Chromosome
        strand = row.Strand
        seq = genome_str.get(chrom)
        if seq is None:
            continue
        donor_seq, acceptor_seq = find_gt_ag(seq, strand, start, end, flank_short, flank_long)
        # Check if the sequences are of the expected length and contain GT/AG
        if donor_seq is not None and acceptor_seq is not None:
            if (len(donor_seq) != expected_length or len(acceptor_seq) != expected_length):
                continue
            if (donor_seq[flank_short:flank_short+2] != "GT" or acceptor_seq[flank_long:flank_long+2] != "AG"):
                continue
            sequences.append(donor_seq + acceptor_seq)
    return sequences


def reverse_complement(seq):
    """Returns the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGT", "TGCA")
    return seq.translate(complement)[::-1]


def find_gt_ag(seq, strand, start, end, flank_short, flank_long):
    """
    Extracts 35bp sequences for splice sites.

    For the positive strand:
      - Donor: 3bp upstream + 'GT' + 30bp downstream.
      - Acceptor: 30bp upstream + 'AG' + 3bp downstream.
    For the negative strand, the corresponding regions are extracted (with appropriate offsets)
    and then reverse complemented so that the final sequences are in 5′–3′ order.
    """
    expected_length = flank_short + 2 + flank_long  # should be 35 bp

    if strand == '+':
        # Check that donor motif is 'GT' and acceptor motif is 'AG'
        if seq[start:start+2] != 'GT':
            return None, None
        if seq[end-2:end] != 'AG':
            return None, None

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

        return donor_seq, acceptor_seq

    else:  # Negative strand
        # On the negative strand the genomic sequence holds the reverse complement of the desired motifs.
        # Therefore, we expect 'AC' at the donor site (reverse complement of GT)
        # and 'CT' at the acceptor site (reverse complement of AG).
        if seq[end-2:end] != 'AC':
            return None, None
        if seq[start:start+2] != 'CT':
            return None, None

        # For negative strand, the donor site is at the end coordinate.
        # Extract from (end - 2 - flank_long) to (end + flank_short) and reverse complement.
        donor_raw = seq[max(0, end - 2 - flank_long): end + flank_short]
        donor_seq = reverse_complement(donor_raw)

        # For negative strand, the acceptor site is at the start coordinate.
        # Extract from (start - flank_short) to (start + 2 + flank_long) and reverse complement.
        acceptor_raw = seq[max(0, start - flank_short): start + 2 + flank_long]
        acceptor_seq = reverse_complement(acceptor_raw)

        return donor_seq, acceptor_seq


# Find the nearest downstream occurrence of motif within a limited window
def find_downstream(seq, start, motif, max_distance=1000):
    search_seq = seq[start:start + max_distance]  # Slice the search region
    match = re.search(motif, search_seq)
    return start + match.start() if match else None


# Find the nearest upstream occurrence of motif within a limited window
def find_upstream(seq, start, motif, max_distance=1000):
    search_start = max(0, start - max_distance)  # Prevent negative index
    search_seq = seq[search_start:start]  # Slice the search region
    matches = list(re.finditer(motif, search_seq))
    return (search_start + matches[-1].start()) if matches else None


def generate_eazy_negatives_introns(introns, genome_fasta):
    output_data = []
    # Extract sequences from the genome FASTA file
    genome_dict = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))
    genome_str  = {
        chrom: str(rec.seq).upper()
        for chrom, rec in genome_dict.items()
    }
    genome_dict.clear()
    
    for row in introns.itertuples(index=True, name="Junction"):
        start = row.Start
        end = row.End
        chrom = row.Chromosome
        strand = row.Strand
        Junction_ID = row.Junction_ID
        Score = row.Score
        seq = genome_str.get(chrom)
        if seq is None:
            continue
        if strand == '+':
            new_start = find_upstream(seq, start, "CT", 2000)
            if new_start is not None:
                new_end = find_downstream(seq, new_start + 100, "AC", 2000)
                if new_end is not None:
                    output_data.append({
                        'Chromosome': chrom,
                        'Start': new_start,
                        'End': new_end + 2,
                        'Junction_ID': Junction_ID,
                        'Score': Score,
                        'Strand': '-'
                    })
        # if strand == '+':
        #     new_end = find_downstream(seq, end, "AC", 2000)
        #     if new_end is not None:
        #         new_start = find_upstream(seq, new_end - 100, "CT", 2000)
        #         if new_start is not None:
        #             output_data.append({
        #                 'Chromosome': chrom,
        #                 'Start': new_start,
        #                 'End': new_end + 2,
        #                 'Junction_ID': Junction_ID,
        #                 'Score': Score,
        #                 'Strand': '-'
        #             })
        else:  # strand == '-'
            new_start = find_upstream(seq, start, "GT", 2000)
            if new_start is not None:
                new_end = find_downstream(seq, new_start + 100, "AG", 2000)
                if new_end is not None:
                    output_data.append({
                        'Chromosome': chrom,
                        'Start': new_start,
                        'End': new_end + 2,
                        'Junction_ID': Junction_ID,
                        'Score': Score,
                        'Strand': '+'
                    })

    return pd.DataFrame(output_data)


def one_hot_encode(seq, seq_length):
    """One-hot encode a DNA sequence to a shape (4, seq_length)."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded = np.zeros((4, seq_length), dtype=np.float32)

    for i, base in enumerate(seq[:seq_length]):  # Trim to fixed length
        encoded[:, i] = mapping.get(base, [0, 0, 0, 0])  # Handle invalid bases (N, etc.)

    return encoded


class SpliceSiteDataset(Dataset):
    def __init__(self, positive_seqs, negative_seqs, seq_length):
        # Encode sequences
        pos_encoded = [one_hot_encode(seq, seq_length) for seq in positive_seqs]
        neg_encoded = [one_hot_encode(seq, seq_length) for seq in negative_seqs]
        
        # Combine data
        all_encoded = np.array(pos_encoded + neg_encoded, dtype=np.float32)
        self.sequences = torch.from_numpy(all_encoded)
        self.labels = torch.tensor([1] * len(pos_encoded) + [0] * len(neg_encoded), dtype=torch.float32)  # 1 for positive, 0 for negative
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


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

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)  # Fix label shape to match output
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def random_split(sequences, test_size=0.2, seed=1053):
    random.seed(seed)
    sequences_shuffled = sequences.copy()
    random.shuffle(sequences_shuffled)
    
    split_idx = int(len(sequences_shuffled) * (1 - test_size))
    train_seqs = sequences_shuffled[:split_idx]
    test_seqs = sequences_shuffled[split_idx:]
    
    return train_seqs, test_seqs


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Train splicing model with input junctions and reference genome")
    parser.add_argument("--reliable_junctions", type=str, required=True,
                        help="Path to reliable junctions BED file")
    parser.add_argument("--out_junctions", type=str, required=True,
                        help="Path to out junctions BED file")
    parser.add_argument("--reference_genome", type=str, required=True,
                        help="Path to reference genome FASTA file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file name (species-specific)")

    args = parser.parse_args()

    reliable_bed = args.reliable_junctions
    tiecov_bed = args.out_junctions
    fasta_file = args.reference_genome
    output_file = args.output_file
    
    # Example: Print the arguments to confirm
    print("Reliable junctions file:", args.reliable_junctions)
    print("Out junctions file:", args.out_junctions)
    print("Reference genome file:", args.reference_genome)
    print("Output file name:", args.output_file)

    flank_short = 3  # Short flank length
    flank_long = 35  # Long flank length
    seq_length = 2 * (flank_short + 2 + flank_long)  # Total sequence length for each splice site

    # Generate positive and negative introns
    positive_introns = generate_positive_introns(reliable_bed)
    eazy_negatives_introns = generate_eazy_negatives_introns(positive_introns, fasta_file)
    difficult_negatives_introns = generate_more_difficult_negatives_introns(tiecov_bed)

    # Extract sequences from the introns
    positives = extract_sequences_from_introns(positive_introns, fasta_file, flank_short, flank_long)
    eazy_negatives = extract_sequences_from_introns(eazy_negatives_introns, fasta_file, flank_short, flank_long)
    more_difficult_negatives = extract_sequences_from_introns(difficult_negatives_introns, fasta_file, flank_short, flank_long)

    # Combine negatives
    negatives = eazy_negatives + more_difficult_negatives

    # Create datasets
    train_pos, test_pos = random_split(positives)
    train_neg, test_neg = random_split(negatives)
    train_dataset = SpliceSiteDataset(train_pos, train_neg, seq_length)
    test_dataset = SpliceSiteDataset(test_pos, test_neg, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    model = SpliceSiteCNN(seq_length).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=30)

    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), output_file)
    print(f"Model saved to {output_file}")


    elapsed_time = time.time() - start_time
    print(f"Total run time: {elapsed_time:.2f} seconds")
    
if __name__ == "__main__":
    main()
