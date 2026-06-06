"""Neuro-Paging live dashboard.

A thin Streamlit view over the MemoryAgent pipeline. Two tabs:

  1. Query Playground - type a query, watch memories rank across tiers with
     a per-hit breakdown of the three scoring terms (semantic / context /
     frequency). Two memories with identical text rank differently by context.

  2. Tier Visualizer - L1 / L2 / L3 occupancy, budgets, and utilization.

Run:  streamlit run ui/dashboard.py

The dashboard adds NO business logic - it only calls agent.remember() /
agent.recall() / get_stats(). All ranking, embedding, tiering is the system.
"""

from __future__ import annotations

import random

import streamlit as st

from neuro_paging.context import ContextTags, TimeBucket
from neuro_paging.embed import BGESmallEmbedder
from neuro_paging.memory.types import Tier
from neuro_paging.pipeline import MemoryAgent
from neuro_paging.routing import ContextAwareScorer

st.set_page_config(page_title="Neuro-Paging", page_icon="🧠", layout="wide")


# (text, time_bucket, foreground_app, semantic_tags)
_SEED = [
    (
        "The CI deployment pipeline broke on the merge to main",
        TimeBucket.MORNING,
        "VSCode",
        ("code", "ci"),
    ),
    # Same text, different context - the headline demo of context-awareness:
    ("The CI deployment pipeline broke on the merge to main", TimeBucket.NIGHT, "Netflix", ()),
    (
        "The user is allergic to peanuts and tree nuts",
        TimeBucket.EVENING,
        "Notes",
        ("health", "food"),
    ),
    ("User prefers their coffee black, no sugar", TimeBucket.MORNING, "Notes", ("food",)),
    (
        "Standup is at 9am every weekday over Zoom",
        TimeBucket.MORNING,
        "Calendar",
        ("work", "routine"),
    ),
    (
        "The user is learning Rust for systems programming",
        TimeBucket.EVENING,
        "VSCode",
        ("code", "learning"),
    ),
    ("Mom's birthday is on March 15th", TimeBucket.AFTERNOON, "Calendar", ("personal",)),
    (
        "The production database runs PostgreSQL 16 with WAL archiving",
        TimeBucket.AFTERNOON,
        "VSCode",
        ("code", "infra"),
    ),
    (
        "User wants window seats on flights, aisle is fine for short trips",
        TimeBucket.EVENING,
        "Notes",
        ("travel", "preference"),
    ),
    (
        "The team uses GitHub Actions for CI and deploys via Docker",
        TimeBucket.MORNING,
        "VSCode",
        ("code", "ci", "infra"),
    ),
]


def _bulk_seed_specs() -> list[tuple[str, TimeBucket, str, tuple[str, ...]]]:
    """Varied, realistic memories so tiers fill like a real user's store.
    Deterministic (seeded) so the demo is reproducible."""
    rng = random.Random(42)
    apps = ["VSCode", "Notes", "Calendar", "Slack", "Mail", "Chrome", "Terminal"]
    buckets = list(TimeBucket)
    lines = [
        "User prefers tabs over spaces in code",
        "User prefers dark mode in every app",
        "The auth service runs on port 8080 behind nginx",
        "The billing service is an AWS Lambda triggered nightly",
        "Search is backed by Elasticsearch with a 7-day retention",
        "The cache layer uses Redis with a 60-second TTL",
        "Weekly design sync is Tuesday at 2pm with the product team",
        "Priya owns the API migration; ping her before schema changes",
        "User is allergic to shellfish - flag it on restaurant bookings",
        "Back up the database before every production deploy",
        "The staging server lives in us-east-1, prod in eu-west-1",
        "User's manager is Devin; 1:1s are Thursday mornings",
        "The mobile app ships every other Wednesday",
        "Renew the TLS certificate before it expires on the 28th",
        "User takes the 8:15 train and works from home on Fridays",
        "The analytics pipeline is Kafka into Snowflake",
        "Feature flags are managed in LaunchDarkly, not in code",
        "User prefers async standups in Slack over live calls",
        "The design system tokens live in the shared Figma file",
        "Incident retros happen the Monday after any Sev-1",
        "User is vegetarian and avoids dairy when possible",
        "The data team owns the dbt models; don't edit them directly",
        "Quarterly planning is the last week of every quarter",
        "User's preferred IDE theme is Solarized Dark",
        "The load balancer health-check path is /healthz",
        "Onboarding docs are in the Notion workspace under People Ops",
        "User commutes by bike when the weather is above 15 degrees",
        "The recommender model retrains every Sunday at midnight",
        "Customer escalations route to the on-call via PagerDuty",
        "User prefers window seats, aisle only for red-eyes",
    ]
    specs: list[tuple[str, TimeBucket, str, tuple[str, ...]]] = []
    for i in range(140):
        text = lines[i % len(lines)]
        if i >= len(lines):
            text = f"{text} (note {i // len(lines)})"
        specs.append((text, rng.choice(buckets), rng.choice(apps), ()))
    return specs


@st.cache_resource(show_spinner=False)
def get_agent() -> MemoryAgent:
    """Build the agent once per session (model load is ~once) and seed it."""
    embedder = BGESmallEmbedder()
    scorer = ContextAwareScorer(embedder=embedder)
    agent = MemoryAgent(data_dir="./.dashboard-memory", embedder=embedder, scorer=scorer)
    if agent.get_stats().total_count == 0:
        # Story memories first (the identical-text pair powers the demo).
        for text, tb, app, tags in _SEED:
            ctx = ContextTags.now(time_bucket=tb, foreground_app=app, semantic_tags=tags)
            agent.remember(text, ctx)

        # Bulk fill so tiers cascade: L1 fills (~91), overflow -> L2.
        bulk_ids = []
        for text, tb, app, tags in _bulk_seed_specs():
            ctx = ContextTags.now(time_bucket=tb, foreground_app=app, semantic_tags=tags)
            bulk_ids.append(agent.remember(text, ctx))

        # Seed L3 honestly: demote ~18 L2 memories down to L3, via the same
        # demote() the pruner uses. Now all three tiers show non-zero.
        demoted = 0
        for mid in bulk_ids:
            if demoted >= 18:
                break
            mem = agent.manager.get(mid)
            if mem is not None and mem.tier == Tier.L2:
                agent.manager.demote(mid)
                demoted += 1
    return agent


def _score_bar(label: str, value: float, weight: float, color: str) -> str:
    pct = int(round(max(0.0, min(1.0, value)) * 100))
    contribution = value * weight
    return f"""
    <div style="margin:3px 0;">
      <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#666;">
        <span><b>{label}</b> <span style="color:#aaa;">x {weight:.2f}</span></span>
        <span>{value:.3f} -> {contribution:.3f}</span>
      </div>
      <div style="background:#eee;border-radius:4px;height:10px;overflow:hidden;">
        <div style="width:{pct}%;background:{color};height:10px;"></div>
      </div>
    </div>
    """


st.title("🧠 Neuro-Paging")
st.caption(
    "Tiered, context-aware, on-device AI memory. Retrieval ranks by "
    "**semantic similarity x situational context x recency** - not keyword match."
)

with st.spinner("Loading bge-small + seeding ~150 demo memories (first run)..."):
    agent = get_agent()

tab_query, tab_tiers = st.tabs(["🔍 Query Playground", "📊 Tier Visualizer"])


with tab_query:
    st.subheader("Ask the memory system")

    col_q, col_ctx = st.columns([3, 2])
    with col_q:
        query = st.text_input(
            "Query",
            value="how do I fix the deployment pipeline?",
            label_visibility="collapsed",
        )
    with col_ctx:
        cc1, cc2 = st.columns(2)
        with cc1:
            tb = st.selectbox(
                "Time of day",
                [t.value for t in TimeBucket],
                index=[t.value for t in TimeBucket].index(TimeBucket.MORNING.value),
            )
        with cc2:
            app = st.selectbox("Current app", ["VSCode", "Notes", "Calendar", "Netflix", "Zoom"])

    k = st.slider("Top-k", 1, 10, 5)

    if query.strip():
        ctx = ContextTags.now(time_bucket=TimeBucket(tb), foreground_app=app)
        result = agent.recall(query, ctx, k=k)

        st.markdown(
            f"**{len(result.hits)} memories** ranked for *'{query}'* - context: **{tb} - {app}**"
        )

        weights = agent.manager._scorer.weights  # noqa: SLF001
        for rank, hit in enumerate(result.hits, start=1):
            prov = hit.provenance
            tier_color = {"L1": "#16a34a", "L2": "#2563eb", "L3": "#9333ea"}.get(
                prov.tier.value, "#666"
            )
            with st.container(border=True):
                st.markdown(
                    f"**#{rank}** &nbsp; "
                    f"<span style='background:{tier_color};color:white;"
                    f"padding:1px 8px;border-radius:4px;font-size:0.75rem;'>"
                    f"{prov.tier.value}</span> &nbsp; "
                    f"**total {hit.score:.3f}**",
                    unsafe_allow_html=True,
                )
                st.write(hit.text)
                st.markdown(
                    _score_bar("semantic", prov.raw_relevance, weights.alpha, "#2563eb")
                    + _score_bar("context", prov.raw_context_sim, weights.beta, "#16a34a")
                    + _score_bar("frequency", prov.raw_frequency, weights.gamma, "#f59e0b"),
                    unsafe_allow_html=True,
                )

        st.info(
            "Try this: keep the query 'deployment pipeline', set app=VSCode, "
            "time=morning -> note the top hit. Now switch to app=Netflix, "
            "time=night. Two seeded memories have identical text - watch the "
            "context term collapse and the ranking change. Flat RAG would tie them."
        )


with tab_tiers:
    st.subheader("Cache hierarchy")
    stats = agent.get_stats()

    cols = st.columns(3)
    tiers = [
        (
            "L1 - Working",
            stats.l1_count,
            stats.l1_bytes,
            stats.l1_capacity_bytes,
            "in-memory FIFO",
            "#16a34a",
        ),
        (
            "L2 - Hot cache",
            stats.l2_count,
            stats.l2_bytes,
            stats.l2_capacity_bytes,
            "HNSW + SQLite",
            "#2563eb",
        ),
        (
            "L3 - Archive",
            stats.l3_count,
            stats.l3_bytes,
            stats.l3_capacity_bytes,
            "HNSW + SQLite",
            "#9333ea",
        ),
    ]
    for col, (name, count, used, cap, kind, color) in zip(cols, tiers, strict=True):
        util = (used / cap * 100) if cap else 0.0
        with col:
            st.markdown(
                f"<div style='border-top:4px solid {color};padding:8px 0;'>"
                f"<div style='font-size:0.8rem;color:#666;'>{name}</div>"
                f"<div style='font-size:2.4rem;font-weight:700;line-height:1;'>{count}</div>"
                f"<div style='font-size:0.75rem;color:#999;'>{kind}</div>"
                f"<div style='font-size:0.72rem;color:#bbb;margin-top:4px;'>"
                f"{used:,} / {cap:,} B - {util:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        f"**Total: {stats.total_count} memories.** New memories enter L1 and "
        "cascade L1->L2->L3 as byte budgets fill. A power-gated background "
        "pruner demotes cold L2 memories to L3."
    )

    st.subheader("Add a memory")
    new_text = st.text_input("New memory text", key="new_mem")
    if st.button("Remember") and new_text.strip():
        agent.remember(new_text, ContextTags.now())
        st.success(f"Stored: '{new_text}'")
        st.rerun()
