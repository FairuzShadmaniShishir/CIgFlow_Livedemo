# Use Masters_Thesis env for training

import torch
import logging
from nanobody_generator import NanobodyGenerator
from sklearn.model_selection import train_test_split
from transformers import T5EncoderModel, T5Tokenizer
from typing import Dict, List
import pandas as pd
import subprocess
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

torch.cuda.empty_cache()
''''

import torch

# ----------------------------
# 1️⃣ Prepare paired sequences
# ----------------------------
# paired_sequences = "heavy_sequence </s> antigen_sequence"
paired_sequences = [
    ' '.join(h) + ' </s> ' + ' '.join(a)
    for h, a in zip(df["heavy_sequence"], df["antigen_sequence"])
]

print(f"Total sequences to process: {len(paired_sequences)}")

# ----------------------------
# 2️⃣ Load IgT5 tokenizer and model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokeniser = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Exscientia/IgT5")
model = model.to(device)
model.eval()

'''

def main():
    # Sample nanobody sequences (replace with your actual data)
    # For demonstration, you need to provide a list of sequences or load from a DataFrame
    # df = pd.read_csv('/home/f087s426/Research/Nanobody_Thermo_Prediction/processed_protein_sequences.csv')
    # sample_sequences = df['Sequence']  # Assuming df['Sequence'] is defined
    import glob
    parquet_files = glob.glob(
        "/home/f087s426/Research/Antibody Research/Antigen_Specific Antibody Design/ASD_ Antigen Specific Antibody Database-20250813T150937Z-1-001/ASD_ Antigen Specific Antibody Database/asd/*.parquet")
    # Read the Parquet file
    # df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    # filtered_df = df[df['dataset'] == 'buzz']

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    # Step 2: Define the expected dataset names
    #dataset_names = ["alpha_seq", "abbd", "buzz", "covid-19"]
    dataset_names = ["covid-19"]
    # abbd= 7 antigen, buzz=1, alphaseq=4,

    # Step 3: Filter only for known datasets
    df = df[(df["dataset"].isin(dataset_names))]
    #  & (df["scfv"] == False)]

    # Step 4: Keep only relevant columns (and drop rows with missing data)
    required_cols = ["dataset", "heavy_sequence", "antigen_sequence"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in your DataFrame: {missing_cols}")

    df = df[required_cols].dropna().reset_index(drop=True)

    # Step 5: Sort for easy readability
    df = df.sort_values(by="dataset").reset_index(drop=True)

    paired_sequences = [
        ' '.join(h) + ' </s> ' + ' '.join(a)
        for h, a in zip(df["heavy_sequence"], df["antigen_sequence"])
    ]

    print(type(paired_sequences[0]))
    print(type(paired_sequences))
    # Step 6: Quick summary
    print("✅ Combined dataset prepared successfully!")
    print("✅ Total samples:", len(df))
    #print("✅ Datasets included:", df["dataset"].unique())
    #print(df.head())

    #sample_sequences = df['heavy_sequence']  # Assuming df['Sequence'] is defined
    antigen_sequences = df['antigen_sequence']

    # Initialize generator
    generator = NanobodyGenerator(seq_dim=128, max_seq_len=256, hidden_dim=512)

    # Split sequences into training and validation
    train_sequences, val_sequences, train_antigen, val_antigen = train_test_split(paired_sequences[:100], antigen_sequences[:100], test_size=0.1, random_state=42)

    # Prepare training dataset
    flow_loader, decoder_loader, merged_embeddings,antigen_embeddings, targets, X = generator.prepare_dataset(train_sequences,train_antigen, batch_size=32)

    # Prepare validation dataset
    _, val_loader, val_embeddings, val_antigen_embeddings, val_targets, val_X = generator.prepare_dataset(val_sequences,val_antigen, batch_size=32)

    # Initialize models
    #generator.initialize_models(embed_dim=embeddings.shape[1])
    generator.initialize_models(
        merged_embed_dim=merged_embeddings.shape[1],
        antigen_embed_dim=antigen_embeddings.shape[1]
    )

    # Train models
    generator.train_flow_model(flow_loader, merged_embeddings, X, epochs=50)
    generator.train_decoder(decoder_loader, val_loader=val_loader, epochs=50)

    # Generate new sequences
    reference_embedding = merged_embeddings[0]  # Use first sequence as reference
    #print(merged_embeddings[0].shape)
    #print(train_antigen.unique())


    new_sequences = generator.generate_multiple_sequences(
            reference_embedding=reference_embedding,
            antigen_embeddings=antigen_embeddings[0],
            num_sequences=100,
            temperature=0.5,
            num_steps=100,
            noise_scale=1.0,
            guidance_scale=0.5,
            top_k=50,
            top_p=0.9,
            max_len=256,
            batch_size=10,
            show_progress=True,
        )

    print(f"Generated sequences type: {type(new_sequences)}")
    print(f"Number of sequences: {len(new_sequences)}")

    # Convert to dictionary for easier handling
    sequences_dict = {f"Sequence_{i + 1}": seq for i, seq in enumerate(new_sequences)}

    # Print sequence info
    for name, seq in sequences_dict.items():
        print(f"{name}: length {len(seq)}")

    def write_sequences_to_fasta(sequences: Dict[str, str], filename: str):
        """Write sequences to FASTA file."""
        with open(filename, "w") as f:
            for name, seq in sequences.items():
                f.write(f">{name}\n{seq}\n")
        print(f"✓ Wrote {len(sequences)} sequences to {filename}")

    def run_promb_evaluation(fasta_file: str, output_file: str = "scores.csv") -> bool:
        """Run PROMB OASIS evaluation."""
        try:
            cmd = f"promb oasis -o {output_file} {fasta_file}"
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")

            if result.returncode == 0:
                print("✓ PROMB evaluation completed successfully")
                return True
            else:
                print(" PROMB evaluation failed")
                return False

        except Exception as e:
            print(f" Error running PROMB: {e}")
            return False

    def analyze_scores(scores_file: str = "scores.csv"):
        """Analyze PROMB scores and return results."""
        if not os.path.exists(scores_file):
            print(f"{scores_file} not found")
            return None

        try:
            df = pd.read_csv(scores_file)
            print(f"\n=== PROMB OASIS Results ===")
            print(f"Evaluated {len(df)} sequences")
            print("\nScores:")
            print(df.to_string(index=False))

            # Get score column (usually second column)
            score_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            scores = df[score_col]

            print(f"\n=== Statistics ===")
            print(f"Mean: {scores.mean():.4f}")
            print(f"Median: {scores.median():.4f}")
            print(f"Min: {scores.min():.4f}")
            print(f"Max: {scores.max():.4f}")
            print(f"Std: {scores.std():.4f}")

            # Show ranked sequences
            ranked_df = df.sort_values(by=score_col, ascending=False)
            print(f"\n=== Ranked Sequences ===")
            for idx, row in ranked_df.iterrows():
                seq_name = row.iloc[0]
                score = row.iloc[1]
                print(f"{seq_name}: {score:.4f}")

            return df

        except Exception as e:
            print(f" Error analyzing scores: {e}")
            return None

    def filter_top_sequences(scores_df, sequences_dict, top_n=3):
        """Filter and save top N sequences."""
        if scores_df is None:
            return None

        score_col = scores_df.columns[1] if len(scores_df.columns) > 1 else scores_df.columns[0]
        name_col = scores_df.columns[0]

        # Get top N sequences
        top_sequences = scores_df.nlargest(top_n, score_col)
        filtered_dict = {}

        print(f"\n=== Top {top_n} Sequences ===")
        for idx, row in top_sequences.iterrows():
            seq_name = row[name_col]
            score = row[score_col]
            sequence = sequences_dict.get(seq_name, "")
            filtered_dict[seq_name] = sequence
            print(f"{seq_name}: {score:.4f} (length: {len(sequence)})")

        # Save top sequences
        top_filename = f"top_{top_n}_antibodies.fa"
        write_sequences_to_fasta(filtered_dict, top_filename)

        return filtered_dict

    # Execute the workflow
    print("\n" + "=" * 50)
    print("ANTIBODY SEQUENCE EVALUATION WORKFLOW")
    print("=" * 50)

    # Step 1: Write sequences to FASTA
    write_sequences_to_fasta(sequences_dict, "Unguided_antibodies_nanobody_temp_1.2.fa")

    # Step 2: Run PROMB evaluation
    success = run_promb_evaluation("Unguided_antibodies_nanobody_temp_1.2.fa",
                                   "Unguided_antibodies_nanobody_temp_1.2._scores.csv")

    if success:
        # Step 3: Analyze results
        scores_df = analyze_scores("Unguided_antibodies_nanobody_temp_1.2._scores.csv")

        # Step 4: Filter top sequences
        if scores_df is not None:
            top_sequences = filter_top_sequences(scores_df, sequences_dict, top_n=3)

            print(f"\n=== Summary ===")
            print(f"✓ Generated {len(new_sequences)} sequences")
            print(f"✓ Evaluated with PROMB OASIS")
            print(f"✓ Saved top 3 sequences")
            print(f"\nFiles created:")
            print(f"- antibodies.fa (all sequences)")
            print(f"- scores.csv (evaluation results)")
            print(f"- top_3_antibodies.fa (best sequences)")
        else:
            print("Could not analyze scores")
    else:
        print("Evaluation failed - check PROMB installation")

    logger.info("Generated sequences:")
    for i, seq in enumerate(new_sequences):
        logger.info(f"Sequence {i + 1}: {seq}")


if __name__ == "__main__":
    main()