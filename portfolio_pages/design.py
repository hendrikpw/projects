"""Shared editorial design system for the hosted portfolio."""

from __future__ import annotations

import streamlit as st


DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
  --ink: #101319;
  --ink-deep: #050607;
  --surface: #171a20;
  --surface-raised: #20242c;
  --paper: #fcfcfd;
  --text: #fcfcfd;
  --muted: rgba(252, 252, 253, .70);
  --quiet: rgba(252, 252, 253, .46);
  --line: rgba(252, 252, 253, .16);
  --line-strong: rgba(252, 252, 253, .34);
  --signal: #e5484d;
  --ease: cubic-bezier(.4, 0, .2, 1);
}

html { scroll-behavior: smooth; }
html, body, [class*="css"] {
  font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
  font-weight: 400;
}
h1, h2, h3, h4 {
  font-family: "Inter", "Helvetica Neue", Arial, sans-serif !important;
  font-weight: 300 !important;
  letter-spacing: -.045em !important;
}
p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }

[data-testid="stAppViewContainer"] {
  background:
    linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px),
    var(--ink);
  background-size: 72px 72px;
  color: var(--text);
}
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  background:
    radial-gradient(circle at 82% 4%, rgba(229,72,77,.08), transparent 30rem),
    linear-gradient(180deg, rgba(0,0,0,.10), rgba(0,0,0,.34));
}
[data-testid="stHeader"] {
  background: rgba(5, 6, 7, .76);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] {
  background: var(--ink-deep);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  font-size: .95rem;
  font-weight: 500 !important;
  letter-spacing: .08em !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .72rem .75rem;
  border-left: 2px solid transparent;
  transition: color .25s var(--ease), border-color .25s var(--ease), background .25s var(--ease);
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  color: var(--paper);
  background: rgba(255,255,255,.05);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  border-left-color: var(--signal);
  background: rgba(255,255,255,.065);
  color: var(--paper);
}
.block-container {
  max-width: 1480px;
  padding-top: 2.75rem;
  padding-bottom: 7rem;
}

/* Editorial hero */
.editorial-hero {
  position: relative;
  min-height: min(680px, 72vh);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  overflow: hidden;
  padding: clamp(2rem, 5vw, 5rem);
  margin: 0 0 clamp(4rem, 8vw, 8rem);
  border: 1px solid var(--line);
  background:
    linear-gradient(135deg, rgba(255,255,255,.05), transparent 42%),
    radial-gradient(circle at 82% 24%, rgba(229,72,77,.26), transparent 25%),
    linear-gradient(120deg, #161a21 0%, #070809 72%);
}
.editorial-hero::after {
  content: "";
  position: absolute;
  right: -10%;
  bottom: -48%;
  width: min(54vw, 700px);
  aspect-ratio: 1;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 50%;
  box-shadow:
    0 0 0 5vw rgba(255,255,255,.018),
    0 0 0 10vw rgba(255,255,255,.012);
}
.editorial-hero > * { position: relative; z-index: 1; }
.brand-line, .eyebrow, .section-kicker {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .14em;
  font-size: .72rem;
  font-weight: 500;
}
.brand-line {
  display: flex;
  align-items: center;
  gap: .8rem;
}
.brand-line::before {
  content: "";
  display: block;
  width: 2.5rem;
  height: 2px;
  background: var(--signal);
}
.editorial-hero h1 {
  max-width: 980px;
  margin: clamp(3rem, 8vh, 7rem) 0 1.4rem;
  color: var(--paper);
  font-size: clamp(3.9rem, 9vw, 8.8rem);
  line-height: .86;
}
.editorial-hero p {
  max-width: 680px;
  margin: 0;
  font-size: clamp(1rem, 1.45vw, 1.25rem);
  line-height: 1.65;
}
.hero-footer {
  display: flex;
  justify-content: space-between;
  gap: 2rem;
  align-items: flex-end;
  margin-top: 3rem;
  color: var(--quiet);
  font-size: .76rem;
  letter-spacing: .08em;
  text-transform: uppercase;
}

/* Section rhythm */
.section-intro {
  display: grid;
  grid-template-columns: minmax(120px, .45fr) minmax(0, 1.55fr);
  gap: 2rem;
  align-items: start;
  padding: 0 0 2.2rem;
  margin: 0 0 2rem;
  border-bottom: 1px solid var(--line);
}
.section-intro h2 {
  max-width: 900px;
  margin: -.25rem 0 0;
  font-size: clamp(2.2rem, 4.8vw, 5rem);
  line-height: 1.02;
}
.section-kicker { color: var(--quiet); }

/* Project panels */
.project-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1px;
  margin: 0 0 clamp(4rem, 8vw, 8rem);
  background: var(--line);
  border: 1px solid var(--line);
}
.project-card {
  position: relative;
  min-height: 390px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  overflow: hidden;
  padding: clamp(1.7rem, 3vw, 3.2rem);
  background: var(--surface);
  transition:
    background .35s var(--ease),
    transform .35s var(--ease);
}
.project-card::before {
  content: "";
  position: absolute;
  inset: auto -18% -52% auto;
  width: 68%;
  aspect-ratio: 1;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,.12);
  transition: transform .7s var(--ease), border-color .35s var(--ease);
}
.project-card--air::after,
.project-card--jobs::after,
.project-card--energy::after,
.project-card--collision::after {
  content: "";
  position: absolute;
  inset: 0;
  opacity: .44;
  pointer-events: none;
}
.project-card--air::after {
  background: radial-gradient(circle at 90% 95%, rgba(229,72,77,.40), transparent 32%);
}
.project-card--jobs::after {
  background: linear-gradient(135deg, transparent 58%, rgba(255,255,255,.06));
}
.project-card--energy::after {
  background:
    linear-gradient(145deg, transparent 45%, rgba(229,72,77,.12)),
    radial-gradient(circle at 78% 82%, rgba(255,255,255,.08), transparent 26%);
}
.project-card--collision::after {
  background:
    linear-gradient(160deg, transparent 38%, rgba(255,255,255,.055)),
    repeating-radial-gradient(
      circle at 82% 84%,
      rgba(229,72,77,.13) 0 1px,
      transparent 1px 34px
    );
}
.project-card:hover { background: var(--surface-raised); transform: translateY(-4px); }
.project-card:hover::before { transform: scale(1.08); border-color: rgba(255,255,255,.28); }
.project-card > * { position: relative; z-index: 1; }
.project-index {
  color: var(--signal);
  font-size: .8rem;
  letter-spacing: .12em;
}
.project-card h3 {
  max-width: 520px;
  margin: 4.8rem 0 1rem;
  color: var(--paper);
  font-size: clamp(2rem, 3.3vw, 3.5rem);
  line-height: 1.02;
}
.project-card p { max-width: 520px; line-height: 1.65; }
.project-meta {
  display: flex;
  flex-wrap: wrap;
  gap: .65rem 1.1rem;
  margin-top: 2rem;
  color: var(--quiet);
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .09em;
}
.project-meta span:not(:last-child)::after {
  content: "/";
  margin-left: 1.1rem;
  color: var(--signal);
}

.statement-panel {
  padding: clamp(3rem, 8vw, 8rem) 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.statement-panel h2 {
  max-width: 1120px;
  margin: 0;
  color: var(--paper);
  font-size: clamp(2.8rem, 6.5vw, 7rem);
  line-height: .98;
}
.statement-panel strong { color: var(--signal); font-weight: 300; }

/* Project page hero */
.page-hero {
  position: relative;
  overflow: hidden;
  padding: clamp(2rem, 5vw, 5rem);
  margin-bottom: 2.6rem;
  min-height: 340px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  background:
    linear-gradient(115deg, rgba(255,255,255,.045), transparent 40%),
    var(--ink-deep);
  border: 1px solid var(--line);
}
.page-hero::after {
  content: "";
  position: absolute;
  right: -5rem;
  top: -10rem;
  width: 28rem;
  height: 28rem;
  border: 1px solid rgba(229,72,77,.36);
  border-radius: 50%;
  box-shadow: 0 0 0 5rem rgba(255,255,255,.018);
}
.page-hero > * { position: relative; z-index: 1; }
.page-hero h1 {
  max-width: 1050px;
  margin: 2.5rem 0 1rem;
  color: var(--paper);
  font-size: clamp(3rem, 7vw, 7rem);
  line-height: .92;
}
.page-hero p { max-width: 760px; margin: 0; font-size: 1.05rem; line-height: 1.65; }
.source-line {
  display: inline-flex;
  align-items: center;
  gap: .6rem;
  margin-top: 1.5rem;
  color: var(--quiet);
  font-size: .72rem;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.source-line::before { content: ""; width: 1.8rem; height: 1px; background: var(--signal); }

/* Streamlit-native components */
[data-testid="stMetric"] {
  min-height: 142px;
  background: rgba(23,26,32,.88);
  border: 1px solid var(--line);
  border-top: 2px solid var(--paper);
  border-radius: 0;
  padding: 1.15rem 1.2rem;
  transition: background .35s var(--ease), transform .35s var(--ease), border-color .35s var(--ease);
}
[data-testid="stMetric"]:hover {
  background: var(--surface-raised);
  border-top-color: var(--signal);
  transform: translateY(-3px);
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] {
  color: var(--paper);
  font-weight: 300;
  letter-spacing: -.04em;
}
[data-testid="stMetricDelta"] { color: var(--quiet); }

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--line);
  background: rgba(5,6,7,.42);
}
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input {
  background: var(--surface) !important;
  border-color: var(--line-strong) !important;
  border-radius: 0 !important;
}
[data-baseweb="tag"] {
  background: rgba(255,255,255,.10) !important;
  border-radius: 0 !important;
}
.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
  min-height: 3rem;
  border: 1px solid var(--paper) !important;
  border-radius: 0 !important;
  background: var(--paper) !important;
  color: var(--ink-deep) !important;
  font-weight: 500 !important;
  transition:
    background .35s var(--ease),
    color .35s var(--ease),
    transform .35s var(--ease) !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > a:hover {
  background: transparent !important;
  color: var(--paper) !important;
  transform: translateY(-2px);
}
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 0 !important;
  background: rgba(5,6,7,.34);
}
hr { border-color: var(--line) !important; }

/* Audi-inspired 250–350 ms motion and scroll entry */
@keyframes pageEnter {
  from { opacity: 0; transform: translateY(24px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes viewportReveal {
  from { opacity: .15; transform: translateY(52px) scale(.985); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
.editorial-hero, .page-hero {
  animation: pageEnter .7s var(--ease) both;
}
.project-card {
  animation: pageEnter .7s var(--ease) both;
}
.project-card:nth-child(2) { animation-delay: .09s; }
.project-card:nth-child(3) { animation-delay: .18s; }
.project-card:nth-child(4) { animation-delay: .27s; }

@media (prefers-reduced-motion: no-preference) {
  @supports (animation-timeline: view()) {
    [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"],
    .section-intro,
    .statement-panel {
      animation: viewportReveal linear both;
      animation-timeline: view();
      animation-range: entry 4% cover 28%;
    }
    .editorial-hero, .page-hero {
      animation: pageEnter .7s var(--ease) both;
      animation-timeline: auto;
    }
  }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
}

@media (max-width: 780px) {
  .block-container { padding-top: 1.4rem; padding-left: 1rem; padding-right: 1rem; }
  .editorial-hero { min-height: 560px; padding: 1.6rem; }
  .editorial-hero h1 { font-size: clamp(3.5rem, 18vw, 6rem); }
  .hero-footer { flex-direction: column; align-items: flex-start; }
  .section-intro { grid-template-columns: 1fr; gap: 1rem; }
  .project-grid { grid-template-columns: 1fr; }
  .project-card { min-height: 340px; }
  .page-hero { padding: 1.6rem; min-height: 390px; }
}
</style>
"""


def inject_design_system() -> None:
    """Install the shared CSS in the current Streamlit page."""
    st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)
