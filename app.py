import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from nanobody_generator import NanobodyGenerator

MODEL_DIR = "Saved_Model/temperature_0.7"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CiGFlow",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

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


# --------------------------------------------------
# GENERATION
# --------------------------------------------------

def generate_antibodies(
        antibody_seq,
        antigen_seq,
        num_sequences,
        temperature,
        top_k,
        top_p,
        guidance_scale,
        num_steps
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
            num_steps=num_steps,
            noise_scale=1.0,
            guidance_scale=guidance_scale,
            top_k=top_k,
            top_p=top_p,
            max_len=256,
            batch_size=5,
            show_progress=False,
        )

    return sequences


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
# 🧬 CiGFlow

### Antigen-Specific Antibody Design Platform

Generate therapeutic antibody candidates using
Conditional Flow Matching and Protein Language Models.
""")

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("Generation Settings")

    num_sequences = st.slider(
        "Number of Sequences",
        1,
        500,
        100
    )

    temperature = st.slider(
        "Temperature",
        0.1,
        2.0,
        0.7,
        0.1
    )

    top_k = st.slider(
        "Top-K",
        10,
        100,
        50
    )

    top_p = st.slider(
        "Top-P",
        0.50,
        1.00,
        0.90,
        0.05
    )

    guidance_scale = st.slider(
        "Guidance Scale",
        0.0,
        10.0,
        0.0,
        0.5
    )

    num_steps = st.slider(
        "Flow Steps",
        10,
        200,
        100
    )

    generate_btn = st.button(
        "🚀 Generate Antibodies",
        use_container_width=True
    )

# --------------------------------------------------
# EXAMPLES
# --------------------------------------------------

example = st.selectbox(
    "Load Example Dataset",
    [
        "Custom Input",
        "HER2 - Trastuzumab"
    ]
)

if example == "HER2 - Trastuzumab":

    antibody_default = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    )

    antigen_default = (
        "MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLAECRHFDELLVTQNPCTYKITGMAIAIPCINCTGQPILDREAFRIRHPKTPSVQLVHYQMRPGPIPAGPGDRDDNPHISGGSTIYNPNYPNLISSVLYNLVTDLDLWMDPETKDEIQQKIGFGKDSQISVTPEGTSAATYLKSCSWLDSGDVNRQFMQRLIKQLTNAGKLDMISQRLNQKNLQYLREQLARRKHSDLIPEGHEQKLISEEDL"
    )

else:
    antibody_default = ""
    antigen_default = ""

# --------------------------------------------------
# INPUTS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    antibody_seq = st.text_area(
        "Reference Antibody Sequence",
        value=antibody_default,
        height=220
    )

with col2:

    antigen_seq = st.text_area(
        "Antigen Sequence",
        value=antigen_default,
        height=220
    )

# --------------------------------------------------
# INPUT STATS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Antibody Length",
        len(antibody_seq.strip())
    )

with col2:
    st.metric(
        "Antigen Length",
        len(antigen_seq.strip())
    )

# --------------------------------------------------
# GENERATE
# --------------------------------------------------

if generate_btn:

    if len(antibody_seq.strip()) == 0:
        st.error("Reference antibody sequence missing.")
        st.stop()

    if len(antigen_seq.strip()) == 0:
        st.error("Antigen sequence missing.")
        st.stop()

    with st.spinner("Generating antibody candidates..."):

        sequences = generate_antibodies(
            antibody_seq.strip(),
            antigen_seq.strip(),
            num_sequences,
            temperature,
            top_k,
            top_p,
            guidance_scale,
            num_steps
        )

    st.success(
        f"Successfully generated {len(sequences)} candidate antibodies."
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    df = pd.DataFrame({
        "Sequence": sequences,
        "Length": [len(x) for x in sequences]
    })

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Generated",
        len(df)
    )

    col2.metric(
        "Average Length",
        round(df["Length"].mean(), 1)
    )

    col3.metric(
        "Unique Sequences",
        df["Sequence"].nunique()
    )

    st.divider()

    # -----------------------------------------
    # HISTOGRAM
    # -----------------------------------------

    st.subheader("Sequence Length Distribution")

    fig, ax = plt.subplots()

    ax.hist(df["Length"], bins=20)

    ax.set_xlabel("Length")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # -----------------------------------------
    # TABLE
    # -----------------------------------------

    st.subheader("Generated Antibody Candidates")

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    # -----------------------------------------
    # VIEWER
    # -----------------------------------------

    st.subheader("Sequence Viewer")

    idx = st.selectbox(
        "Select Candidate",
        range(len(sequences)),
        format_func=lambda x: f"Candidate {x+1}"
    )

    st.code(
        sequences[idx],
        language=None
    )

    # -----------------------------------------
    # FASTA
    # -----------------------------------------

    fasta_text = ""

    for i, seq in enumerate(sequences):

        fasta_text += f">Candidate_{i+1}\n"

        for j in range(0, len(seq), 80):
            fasta_text += seq[j:j+80] + "\n"

    # -----------------------------------------
    # DOWNLOADS
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "⬇ Download FASTA",
            fasta_text,
            "generated_antibodies.fasta",
            mime="text/plain",
            use_container_width=True
        )

    with col2:

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "generated_antibodies.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()

st.caption(
    "CiGFlow • Conditional Flow Matching for Antigen-Specific Antibody Design"
)
