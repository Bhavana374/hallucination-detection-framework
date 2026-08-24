"""
Streamlit Conversational UI & Hallucination Detector Application.

Supports two modes:
  Mode A: Generate + Analyze (requires LLM API key in .env)
  Mode B: Analyze Existing Response (paste any AI response directly)
"""

import streamlit as st
from src.pipeline.hallucination_detector import HallucinationDetector
from src.utils.llm_provider import get_llm_provider

st.set_page_config(
    page_title="GenAI Hallucination Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .verdict-supported {
        color: #10B981;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 1.15rem;
    }
    .verdict-hallucinated {
        color: #EF4444;
        font-weight: 700;
        background: rgba(239, 68, 68, 0.15);
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 1.15rem;
    }
    .verdict-insufficient {
        color: #F59E0B;
        font-weight: 700;
        background: rgba(245, 158, 11, 0.15);
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 1.15rem;
    }
    .score-display {
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_backend():
    detector = HallucinationDetector()
    llm = get_llm_provider()
    return detector, llm

detector, llm_provider = load_backend()


# --- Header ---
st.markdown('<div class="main-title">🔬 GenAI Hallucination Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Evidence-Grounded Hybrid Transformer Framework · Claim-Level Verification</div>', unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Mode B: Analyze Existing Response", "Mode A: Generate + Analyze"],
    index=0,
    help="Mode A requires an LLM API key configured in .env"
)

preset_choice = st.sidebar.selectbox("Load Example Scenario", [
    "Custom Input",
    "Example 1: Factual (Apollo 11)",
    "Example 2: Contradicted (Fabricated Mars Landing)",
    "Example 3: Mixed (Factual + Hallucinated)",
    "Example 4: Insufficient Evidence"
])

# --- Preset defaults ---
default_prompt = ""
default_resp = ""
default_ref = ""

if preset_choice == "Example 1: Factual (Apollo 11)":
    default_prompt = "Who were the astronauts on Apollo 11 and when did they land?"
    default_resp = "Apollo 11 landed Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969."
    default_ref = "Apollo 11 landed humans on the Moon on July 20, 1969. The astronauts were Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin."
elif preset_choice == "Example 2: Contradicted (Fabricated Mars Landing)":
    default_prompt = "Where did Apollo 11 land?"
    default_resp = "Apollo 11 landed on Mars in October 1975. Commander Yuri Gagarin walked on Mars."
    default_ref = "Apollo 11 landed humans on the Moon on July 20, 1969. The astronauts were Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin."
elif preset_choice == "Example 3: Mixed (Factual + Hallucinated)":
    default_prompt = "Tell me about Apollo 11."
    default_resp = "Apollo 11 landed on the Moon in July 1969. Commander Yuri Gagarin walked on Mars."
    default_ref = "Apollo 11 landed humans on the Moon on July 20, 1969. The astronauts were Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin."
elif preset_choice == "Example 4: Insufficient Evidence":
    default_prompt = "How do quantum computers work?"
    default_resp = "Quantum computers use qubits in superposition for exponential parallel processing."
    default_ref = "The Eiffel Tower is located in Paris, France."
else:
    default_prompt = "Enter your question here..."
    default_resp = "Paste the AI-generated response here..."
    default_ref = "Paste the reference ground truth context here..."


# =====================================================
# MODE A: Generate + Analyze
# =====================================================
if "Mode A" in analysis_mode:
    st.markdown("### Mode A: Generate + Analyze")
    st.caption("Enter a prompt. The configured LLM will generate a response, then the detector analyzes it.")

    user_prompt = st.text_input("Enter Prompt", value=default_prompt if default_prompt != "Enter your question here..." else "")
    user_context = st.text_area("Reference Ground Truth Context", value=default_ref if default_ref != "Paste the reference ground truth context here..." else "", height=120)

    do_generate = st.button("🤖 Generate & Analyze", type="primary", use_container_width=True)


# =====================================================
# MODE B: Analyze Existing Response
# =====================================================
else:
    st.markdown("### Mode B: Analyze Existing Response")
    st.caption("Paste any AI-generated response and reference context. No LLM API key required.")

    user_prompt = st.text_input("Original Question / Prompt (for context)", value=default_prompt if default_prompt != "Enter your question here..." else "")

    col1, col2 = st.columns(2)
    with col1:
        user_response = st.text_area("AI-Generated Response", value=default_resp if default_resp != "Paste the AI-generated response here..." else "", height=160)
    with col2:
        user_context = st.text_area("Reference Ground Truth Context", value=default_ref if default_ref != "Paste the reference ground truth context here..." else "", height=160)

    do_analyze = st.button("🔬 Analyze Response", type="primary", use_container_width=True)


# =====================================================
# SHARED ANALYSIS EXECUTION & RENDERING
# =====================================================
def render_results(results: dict):
    """Render the full hallucination analysis results."""

    st.markdown("---")
    st.markdown("## 💬 AI Response")
    st.info(results["generated_response"])

    # --- Response-Level Summary ---
    st.markdown("## 📊 Hallucination Analysis")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Claims", results["total_claims"])
    with m2:
        st.metric("✓ Supported", results["supported_claims_count"])
    with m3:
        st.metric("✗ Contradicted", results["contradicted_claims_count"])
    with m4:
        st.metric("⚠ Insufficient", results["insufficient_evidence_claims_count"])
    with m5:
        score_pct = results["overall_hallucination_score"]
        st.metric("Hallucination Score", f"{score_pct:.1%}")

    # --- Overall Verdict ---
    st.write("")
    verdict_str = results["overall_verdict"]
    if verdict_str == "SUPPORTED":
        st.markdown(
            '**Overall Verdict:** <span class="verdict-supported">✓ SUPPORTED</span>',
            unsafe_allow_html=True
        )
    elif verdict_str == "HALLUCINATION_DETECTED":
        st.markdown(
            '**Overall Verdict:** <span class="verdict-hallucinated">✗ HALLUCINATION DETECTED</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '**Overall Verdict:** <span class="verdict-insufficient">⚠ INSUFFICIENT EVIDENCE</span>',
            unsafe_allow_html=True
        )

    # --- Claim Breakdown ---
    st.markdown("---")
    st.markdown("## 📜 Claim Analysis")

    for claim in results["claim_breakdown"]:
        v = claim["verdict"]
        c_text = claim["claim_text"]
        c_id = claim["claim_id"]

        if v == "SUPPORTED":
            badge = "✓ SUPPORTED"
        elif v == "CONTRADICTED":
            badge = "✗ CONTRADICTED"
        else:
            badge = "⚠ INSUFFICIENT EVIDENCE"

        expander_title = f'{c_id}: "{c_text}" — {badge}'

        with st.expander(expander_title, expanded=True):
            # Left: core claim info
            info_col, metrics_col = st.columns([3, 2])

            with info_col:
                st.markdown(f"**Claim Text:** {c_text}")
                st.markdown(f'**Retrieved Evidence:** *"{claim["retrieved_evidence"]}"*')
                st.markdown(f"**Verdict:** **{v}**")
                st.markdown(f"**Explanation:** {claim['explanation']['explanation_summary']}")

            with metrics_col:
                st.markdown("**Scores & Probabilities:**")
                st.markdown(f"- Retrieval Score: `{claim['retrieval_score']:.4f}`")
                st.markdown(f"- Semantic Similarity: `{claim['semantic_similarity']:.4f}`")
                st.markdown(f"- NLI Entailment: `{claim['nli_entailment']:.4f}`")
                st.markdown(f"- NLI Contradiction: `{claim['nli_contradiction']:.4f}`")
                st.markdown(f"- NLI Neutral: `{claim['nli_neutral']:.4f}`")
                st.markdown(f"- Hallucination Probability: `{claim['hallucination_probability']:.2%}`")

    # --- Overall Explanation ---
    st.markdown("---")
    st.markdown("## 📝 Explanation")
    total = results["total_claims"]
    supported = results["supported_claims_count"]
    contradicted = results["contradicted_claims_count"]
    insufficient = results["insufficient_evidence_claims_count"]
    score = results["overall_hallucination_score"]

    explanation_parts = [
        f"The response contained **{total} claim(s)**.",
        f"**{supported}** were supported by evidence,",
        f"**{contradicted}** were contradicted,",
        f"and **{insufficient}** had insufficient evidence.",
        f"The overall hallucination risk score is **{score:.1%}**",
        f"with an overall verdict of **{verdict_str.replace('_', ' ')}**."
    ]
    st.markdown(" ".join(explanation_parts))


# --- Execute Analysis ---
if "Mode A" in analysis_mode:
    if do_generate and user_prompt and user_prompt.strip() and user_context and user_context.strip():
        try:
            generated_response = llm_provider.generate_response(user_prompt)
            st.success("Response generated successfully!")
            with st.spinner("Running hallucination detection pipeline..."):
                results = detector.analyze_response(
                    query=user_prompt,
                    generated_response=generated_response,
                    reference_context=user_context
                )
            render_results(results)
        except RuntimeError as e:
            st.error(str(e))
else:
    if do_analyze:
        if not user_response or not user_response.strip():
            st.warning("Please paste an AI-generated response to analyze.")
        elif not user_context or not user_context.strip():
            st.warning("Please provide reference context for evidence grounding.")
        else:
            with st.spinner("Running hallucination detection pipeline (claim extraction → FAISS retrieval → NLI → 20-feature extraction → hybrid classification)..."):
                results = detector.analyze_response(
                    query=user_prompt if user_prompt else "",
                    generated_response=user_response,
                    reference_context=user_context
                )
            render_results(results)

