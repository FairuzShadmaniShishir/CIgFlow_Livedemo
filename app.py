import streamlit as st
import pandas as pd
import torch
from nanobody_generator import NanobodyGenerator

MODEL_DIR = "Saved_Model/temperature_0.7"


@st.cache_resource
def load_generator():

    config = torch.load(
        f"{MODEL_DIR}/config.pt",
        map_location="cpu"
    )

    generator = NanobodyGenerator(
        seq_dim=config["seq_dim"],
        max_seq_len=config["max_seq_len"],
        hidden_dim=config["hidden_dim"],
    )

    generator.initialize_models(
        merged_embed_dim=config["merged_embed_dim"],
        antigen_embed_dim=config["antigen_embed_dim"],
    )

    generator.flow_model.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/flow_model.pt",
            map_location="cpu"
        )
    )

    generator.decoder.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/decoder.pt",
            map_location="cpu"
        )
    )

    generator.flow_model.eval()
    generator.decoder.eval()

    return generator


def generate_antibodies(
    antibody_seq,
    antigen_seq,
    num_sequences,
    temperature,
):

    generator = load_generator()

    paired = " ".join(antibody_seq)

    _, _, merged_emb, antigen_emb, _, _ = generator.prepare_dataset(
        [paired],
        [antigen_seq],
        batch_size=1
    )

    with torch.no_grad():

        sequences = generator.generate_multiple_sequences(
            reference_embedding=merged_emb[0],
            antigen_embeddings=antigen_emb[0],
            num_sequences=num_sequences,
            temperature=temperature,
            num_steps=100,
            noise_scale=1.0,
            guidance_scale=0,
            top_k=50,
            top_p=0.9,
            max_len=256,
            batch_size=5,
            show_progress=False,
        )

    return sequences


st.set_page_config(
    page_title="Antigen-Specific Antibody Generator",
    layout="wide"
)

st.title("🧬 Antigen-Specific Antibody Generator")

st.markdown(
"""
Generate antibody sequences conditioned on an antigen using
a Flow Matching + Protein Language Model framework.
"""
)

antibody_seq = st.text_area(
    "Reference Antibody Sequence",
    height=150
)

antigen_seq = st.text_area(
    "Antigen Sequence",
    height=250
)

col1, col2 = st.columns(2)

with col1:
    num_sequences = st.slider(
        "Number of Sequences",
        1,
        1000,
        100
    )

with col2:
    temperature = st.slider(
        "Temperature",
        0.1,
        2.0,
        0.7,
        0.1
    )

if st.button("Generate Antibodies"):

    if not antibody_seq or not antigen_seq:
        st.error("Please provide both sequences.")
    else:

        with st.spinner("Generating sequences..."):

            sequences = generate_antibodies(
                antibody_seq,
                antigen_seq,
                num_sequences,
                temperature
            )

        df = pd.DataFrame({
            "Sequence": sequences,
            "Length": [len(x) for x in sequences]
        })

        st.success(f"Generated {len(sequences)} antibodies")

        st.dataframe(
            df,
            use_container_width=True
        )

        fasta_text = ""

        for i, seq in enumerate(sequences):

            fasta_text += f">Sequence_{i+1}\n"

            for j in range(0, len(seq), 80):
                fasta_text += seq[j:j+80] + "\n"

        st.download_button(
            label="Download FASTA",
            data=fasta_text,
            file_name="generated_antibodies.fasta",
            mime="text/plain"
        )
