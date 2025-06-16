import argparse
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import os

def parse_gtf(gtf_file):
    gtf_records = []
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'transcript':
                continue
            chrom = fields[0]
            attributes = fields[8]
            
            transcript_id_match = re.search(r'transcript_id "([^"]+)"', attributes)
            class_code_match = re.search(r'class_code "([^"]+)"', attributes)
            
            if transcript_id_match and class_code_match:
                transcript_id = transcript_id_match.group(1)
                class_code = class_code_match.group(1)
                gtf_records.append((chrom, transcript_id, class_code))

    gtf_df = pd.DataFrame(gtf_records, columns=['chrom', 'Transcripts', 'class_code'])
    return gtf_df


def parse_markov_score(markov_score_file):
    data = []

    with open(markov_score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                transcript_id = parts[0]
                score5 = parts[4]  # Fifth column (0-based index 4)
                data.append((transcript_id, float(score5)))
    return pd.DataFrame(data, columns=['Transcripts', 'Markov_Score'])

def extract_reference_mrna_count(stats_file_path):
    with open(stats_file_path, "r") as f:
        for line in f:
            match = re.search(r"Reference mRNAs\s*:\s*(\d+)", line)
            if match:
                count = int(match.group(1))
                print(f"✅ Reference mRNAs: {count}")
                return count
    print("❌ Reference mRNAs not found.")
    return None


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Train splicing model with input junctions and reference genome")
    parser.add_argument("--score_tsv_path", type=str, nargs='+', required=True,
                    help="Path(s) to TSV file(s) with minimum junction scores")
    parser.add_argument("--markov_score", type=str, required=True,
                        help="Path to markov score file")
    parser.add_argument("--annotated_gtf", type=str, required=True,
                        help="Path to annotated gtf file")
    parser.add_argument("--gffcompare_stats", type=str, required=True,
                        help="Path to gffcompare stats file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file name (species-specific)")

    args = parser.parse_args()

    tsv_dfs = []
    for tsv_file in args.score_tsv_path:
        df = pd.read_csv(tsv_file, sep='\t')
        tsv_dfs.append(df)
    markov_score = args.markov_score
    annotated_gtf = args.annotated_gtf
    gffcompare_stats = args.gffcompare_stats
    output_file = args.output_file
    print(f"Input files: {args.score_tsv_path}, {markov_score}, {annotated_gtf}, {gffcompare_stats}")

 
    
    basename = os.path.basename(annotated_gtf)
    species = basename.split('_')[0]
    print(f"Species: {species}")

    reference_mrna_count = extract_reference_mrna_count(gffcompare_stats)
    
    gtf_df = parse_gtf(annotated_gtf)

    markov_df = parse_markov_score(markov_score)

    merged_markov_df = pd.merge(gtf_df, markov_df, on='Transcripts', how='left')
    merged_markov_df = merged_markov_df.sort_values(by='Markov_Score', ascending=True)
    merged_markov_df['Markov_Score'] = merged_markov_df['Markov_Score'].replace(10000, np.nan)

    precision_lists_cnn = []
    recall_lists_cnn = []
    
    for df in tsv_dfs:
        df = pd.merge(gtf_df, df, on='Transcripts', how='left')
        merged_cnn_df = df.sort_values(by='Junction_Score', ascending=True)
        class_code_cnn = (merged_cnn_df['class_code'] == '=')
        merged_cnn_df['label'] = class_code_cnn.astype(int)
        thresholds_cnn = sorted(merged_cnn_df['Junction_Score'].unique())
        
        precision_list_cnn = []
        recall_list_cnn = []
        y_true_cnn = merged_cnn_df['label'].values
        y_scores_cnn = merged_cnn_df['Junction_Score'].values

        thr_pct   = np.nanpercentile(y_scores_cnn, 20)
        thresholds_cnn = [x for x in thresholds_cnn if x < thr_pct]
        for threshold in thresholds_cnn:
            y_pred_cnn = (y_scores_cnn >= threshold).astype(int)
            TP = np.sum((y_true_cnn == 1) & (y_pred_cnn == 1))
            FP = np.sum((y_true_cnn == 0) & (y_pred_cnn == 1))
            FN = np.sum((y_true_cnn == 1) & (y_pred_cnn == 0))
            precision = TP / (TP + FP)
            recall = TP / reference_mrna_count
            precision_list_cnn.append(precision)
            recall_list_cnn.append(recall)
        
        precision_lists_cnn.append(precision_list_cnn)
        recall_lists_cnn.append(recall_list_cnn)
    

    class_code_markov = (merged_markov_df['class_code'] == '=')
    merged_markov_df['label'] = class_code_markov.astype(int)
    thresholds_markov = sorted(merged_markov_df['Markov_Score'].unique())
    
    precision_list_markov = []
    recall_list_markov = []
    y_true_markov = merged_markov_df['label'].values
    y_scores_markov = merged_markov_df['Markov_Score'].values
    thr_pct = np.nanpercentile(y_scores_markov, 20)
    thresholds_markov = [x for x in thresholds_markov if x < thr_pct]

    for threshold in thresholds_markov:
        y_pred_markov = (y_scores_markov >= threshold).astype(int)
        TP = np.sum((y_true_markov == 1) & (y_pred_markov == 1))
        FP = np.sum((y_true_markov == 0) & (y_pred_markov == 1))
        FN = np.sum((y_true_markov == 1) & (y_pred_markov == 0))
        precision = TP / (TP + FP)
        recall = TP / reference_mrna_count
        precision_list_markov.append(precision)
        recall_list_markov.append(recall)


    plt.figure(figsize=(8, 6))
    plt.plot(recall_list_markov, precision_list_markov, label='Markov Model')
    for i, df in enumerate(tsv_dfs):
        plt.plot(recall_lists_cnn[i], precision_lists_cnn[i], label=f'CNN Model {i+1}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{species} Precision-Recall Curve for filtering transcripts')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    elapsed_time = time.time() - start_time
    print(f"Total run time: {elapsed_time:.2f} seconds")

    
if __name__ == "__main__":
    main()
