"""Lab 03 - Chaotic Text Generator: a test bench for text generators (Streamlit UI).

Run:  streamlit run app.py
"""
import html
import random
import re

import pandas as pd
import streamlit as st

from src.pipeline import HMM, NGramGenerator, coarse, generation_metrics, load_tagged, split_by_document

st.set_page_config(page_title="Chaotic Text Generator", page_icon="🎲", layout="wide")

MODELS = ["HMM, plain states", "HMM, role-split states", "Trigram"]
TEMPERATURES = [0.5, 1.0, 1.5, 2.0, 2.5]
NOTICE = ("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined "
          "relative to this corpus and do not measure the quality of real English.")


# ------------------------------------------------------------------ models (estimated once, by counting)
@st.cache_resource
def get_models():
    docs = load_tagged()
    train, _ = split_by_document(docs)          # split by document BEFORE any counting
    return {
        "train": train,
        "plain": HMM(train),
        "split": HMM(train, split_states=True),
        "ngram": NGramGenerator([w for w, _ in train]),
        "patterns": {" ".join(t) for _, t in train},
        "texts": {" ".join(w) for w, _ in train},
    }


M = get_models()


@st.cache_data(show_spinner=False)
def generate(model, temperature, seed, n, prefix=()):
    """Generate n sentences with one model; return (table, metrics)."""
    rng = random.Random(int(seed))
    rows, samples = [], []
    for i in range(n):
        if model == "Trigram":
            words = M["ngram"].sample_sentence(rng, temperature, prefix=list(prefix))
            tags = M["plain"].viterbi(words)     # the trigram has no hidden tags: tag its output with Viterbi
            hidden = " ".join(tags)
        else:
            hmm = M["split"] if model == "HMM, role-split states" else M["plain"]
            words, states = hmm.sample_with_states(rng, temperature)
            tags = coarse(states)
            hidden = " ".join(states)
        sentence = " ".join(words)
        rows.append({
            "#": i + 1,
            "Sentence": sentence,
            "Hidden tag sequence": hidden,
            "Valid pattern": "✅" if " ".join(tags) in M["patterns"] else "❌",
            "Copied from training": "⚠️ yes" if sentence in M["texts"] else "no",
        })
        samples.append((words, tags))
    metrics = generation_metrics(samples, M["train"], M["plain"])
    return pd.DataFrame(rows), metrics


def pct(x):
    return f"{x:.1%}"


# ------------------------------------------------------------------ control panel
with st.sidebar:
    st.header("Control panel")
    model = st.selectbox("Model", MODELS, index=1)
    temperature = round(st.slider("Temperature", 0.3, 3.0, 1.0, 0.1,
                                  help="Sampling weights are count^(1/T). T < 1 is safer, T > 1 is more varied."), 2)
    seed = int(st.number_input("Random seed", 0, 10_000, 42, step=1))
    n = int(st.slider("Number of sentences", 10, 500, 500, 10))

    prefix = ()
    if model == "Trigram":
        prefix = tuple(st.text_input("Trigram prompt (optional)", "").lower().split())
    bad_prefix = [w for w in prefix if w not in M["ngram"].vocab]
    if bad_prefix:
        st.error(f"Unknown word(s) in the prompt: {', '.join(bad_prefix)}.")

    if st.button("Generate", type="primary", disabled=bool(bad_prefix), use_container_width=True):
        st.session_state.params = (model, temperature, seed, n, prefix)

# ------------------------------------------------------------------ page
st.title("Chaotic Text Generator")
st.caption("A test bench for text generators: part-of-speech HMM (plain or role-split states) against a trigram model.")
st.info(NOTICE, icon="ℹ️")

tab_gen, tab_tag, tab_cmp = st.tabs(["Generate", "Tagger (Viterbi)", "Compare models"])

# ---- 1. generation table + metrics + CSV
with tab_gen:
    if "params" not in st.session_state:
        st.write("Choose the settings in the control panel, then press **Generate**.")
    else:
        p_model, p_T, p_seed, p_n, p_prefix = st.session_state.params
        with st.spinner("Generating…"):
            df, m = generate(*st.session_state.params)

        st.subheader(f"{p_model} · T = {p_T} · seed {p_seed} · {p_n} sentences")
        c = st.columns(4)
        c[0].metric("Valid tag pattern", pct(m["valid_tag_pattern"]),
                    help="Tag sequence identical to one of the 16 sequences of the training set.")
        c[1].metric("Plausible subject-verb", pct(m["plausible_subject_verb"]),
                    help="The (first noun, first verb) pair was seen in the training set.")
        c[2].metric("Novel sentences", pct(m["novel_sentences"]),
                    help="Share of sentences not copied verbatim from the training set.")
        c[3].metric("Distinct-2", f"{m['distinct_2']:.3f}",
                    help="Different word bigrams / total word bigrams (diversity).")

        st.dataframe(df, hide_index=True, height=440)
        st.caption("For the role-split HMM the hidden states show the role (S = before the verb, O = after the verb, "
                   "P = in the prepositional phrase). The trigram has no hidden states, so its sentences are tagged "
                   "with Viterbi.")

        slug = p_model.lower().replace(",", "").replace(" ", "_")
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"generated_{slug}_T{p_T}_seed{p_seed}.csv", mime="text/csv")

# ---- 2. tagger
with tab_tag:
    st.write("Type a sentence: the Viterbi algorithm returns the most probable tag for every word.")
    text = st.text_input("Sentence", "the children watch the birds .")
    use_split = st.checkbox("Tag with the role-split HMM", value=False)
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())[:60]

    if not tokens:
        st.info("Type at least one word.")
    else:
        tagger = M["split"] if use_split else M["plain"]
        try:
            tags = tagger.viterbi(tokens)
        except Exception as e:  # never crash the app
            st.error(f"Tagging failed: {e}")
            tags = []
        unknown = sorted({w for w in tokens if w not in tagger.vocab})
        if unknown:
            st.warning(f"Out-of-vocabulary word(s): **{', '.join(unknown)}**. Their tag is guessed from the "
                       "surrounding tags only (smoothed emission probability), so it may be wrong.", icon="⚠️")
        if tags:
            chips = []
            for w, t in zip(tokens, tags):
                border = "#e8a33d" if w not in tagger.vocab else "rgba(128,128,128,.4)"
                chips.append(
                    f'<div style="display:inline-block;text-align:center;margin:4px 5px;padding:6px 12px;'
                    f'border-radius:8px;border:1.5px solid {border}">'
                    f'<div style="font-weight:600;font-size:1.05em">{html.escape(w)}</div>'
                    f'<div style="font-size:.8em;opacity:.7;letter-spacing:.04em">{html.escape(t)}</div></div>')
            st.markdown("".join(chips), unsafe_allow_html=True)
    with st.expander(f"Training vocabulary ({len(M['plain'].vocab)} words)"):
        st.write(", ".join(sorted(M["plain"].vocab)))

# ---- 3. comparison (optional deliverable)
with tab_cmp:
    st.write(f"The three models at the same settings: T = {temperature}, seed {seed}, {n} sentences.")
    with st.spinner("Comparing models…"):
        rows = []
        for name in MODELS:
            _, mm = generate(name, temperature, seed, n, ())
            rows.append({"Model": name,
                         "Valid tag pattern": pct(mm["valid_tag_pattern"]),
                         "Plausible subject-verb": pct(mm["plausible_subject_verb"]),
                         "Novel sentences": pct(mm["novel_sentences"]),
                         "Distinct-2": f"{mm['distinct_2']:.3f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        st.subheader("Valid tag pattern (%) against temperature")
        curve = {name: [100 * generate(name, t, seed, n, ())[1]["valid_tag_pattern"] for t in TEMPERATURES]
                 for name in MODELS}
        st.line_chart(pd.DataFrame(curve, index=pd.Index(TEMPERATURES, name="Temperature")))
    st.caption("Higher temperature means more diversity and more structural errors. The role-split HMM stays at "
               "100% because its states carry the role memory the plain first-order HMM lacks.")