from __future__ import annotations

import re
import json
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# IMPORTANT:
# Keep bwa_backend.py in the same folder as this file.
# If your backend file has another name, change this import accordingly.
from AIBLOG import app as blog_graph


# ---------------------------------------------------------
# PAGE CONFIG + THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="BlogForge | AI Writing Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0A0F1E;
        --panel: #111A2D;
        --panel-2: #16223A;
        --line: #263653;
        --text: #F2F5FF;
        --muted: #A8B4CE;
        --purple: #9B8CFF;
        --blue: #5CA8FF;
        --mint: #6EE7C2;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(111, 88, 255, .15), transparent 32%),
            radial-gradient(ellipse at 92% 8%, rgba(55, 145, 255, .10), transparent 28%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(10, 15, 30, .88);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111A2D 0%, #0E1729 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -.4px;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .hero {
        padding: 1.5rem 0 1.2rem 0;
    }

    .eyebrow {
        color: var(--purple);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #F5F6FF;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.08;
        font-weight: 700;
        letter-spacing: -1.8px;
        margin: 0;
    }

    .hero-title span {
        background: linear-gradient(90deg, #A99BFF, #65B5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #AAB6D0;
        font-size: 15px;
        margin-top: 12px;
        max-width: 720px;
    }

    .soft-card {
        background: linear-gradient(145deg, rgba(23, 35, 59, .96), rgba(16, 26, 45, .96));
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 18px 20px;
        margin: 8px 0 14px 0;
    }

    .soft-card-title {
        color: #F3F5FF;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .soft-card-sub {
        color: #9EABC5;
        font-size: 13px;
    }

    .mini-label {
        color: #9D91FF;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 10px;
        font-weight: 800;
    }

    .stTextArea textarea, .stTextInput input {
        background: #101B30 !important;
        color: #F4F6FF !important;
        border: 1px solid #304260 !important;
        border-radius: 12px !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #9687FF !important;
        box-shadow: 0 0 0 1px rgba(150, 135, 255, .2) !important;
    }

    [data-baseweb="select"] > div {
        background: #101B30;
        border-color: #304260;
        border-radius: 10px;
        color: #F4F6FF;
    }

    .stDateInput input {
        background: #101B30 !important;
        color: #F4F6FF !important;
        border-radius: 10px !important;
        border-color: #304260 !important;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 42px;
        transition: all .18s ease;
    }

    .stButton > button[kind="primary"] {
        color: white !important;
        border: 0 !important;
        background: linear-gradient(100deg, #7665F5, #398FEA) !important;
        box-shadow: 0 7px 22px rgba(102, 99, 245, .22);
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.12);
        transform: translateY(-1px);
    }

    .stButton > button:not([kind="primary"]), .stDownloadButton > button {
        background: #172641 !important;
        color: #DCE6FF !important;
        border: 1px solid #35496B !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #293956;
    }

    .stTabs [data-baseweb="tab"] {
        color: #AAB6D0;
        background: transparent;
        border-radius: 10px 10px 0 0;
        padding: 11px 15px;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: #B5A9FF !important;
        border-bottom: 2px solid #9B8CFF !important;
    }

    [data-testid="stMetric"] {
        background: #131F35;
        border: 1px solid #2B3D5B;
        padding: 14px 16px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #A8B4CE !important;
    }

    [data-testid="stMetricValue"] {
        color: #F2F5FF !important;
    }

    [data-testid="stExpander"] {
        background: #111C30;
        border: 1px solid #2B3D5B;
        border-radius: 12px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #293956;
    }

    code {
        color: #C4B9FF !important;
    }

    .status-pill {
        display: inline-block;
        border: 1px solid rgba(110, 231, 194, .35);
        background: rgba(110, 231, 194, .08);
        color: #83EBCB;
        border-radius: 99px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 700;
    }

    .empty-state {
        border: 1px dashed #344766;
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        color: #A8B4CE;
        background: rgba(18, 29, 49, .55);
    }

    .empty-state strong {
        color: #F0F3FF;
        font-size: 17px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# PATHS + SESSION STATE
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
BLOGS_DIR = BASE_DIR / "generated_blogs"
IMAGES_DIR = BASE_DIR / "images"
BLOGS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

for key, default in {
    "result": None,
    "logs": [],
    "last_topic": "",
    "loaded_blog": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def safe_slug(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", str(value), flags=re.UNICODE).strip().lower()
    value = re.sub(r"[\s-]+", "_", value)
    return value[:90] or "generated_blog"


def as_plain_dict(value: Any) -> dict:
    """Convert Pydantic models or mapping-like values into a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def markdown_image_path(md_text: str) -> str:
    """Resolve relative image paths against this app's folder."""
    def replace_image(match: re.Match) -> str:
        alt, raw_path = match.group(1), match.group(2)
        raw_path = raw_path.strip()
        if raw_path.startswith(("http://", "https://", "data:")):
            return match.group(0)
        candidate = (BASE_DIR / raw_path).resolve()
        try:
            relative = candidate.relative_to(BASE_DIR.resolve())
        except ValueError:
            return match.group(0)
        if candidate.exists():
            return f"![{alt}]({relative.as_posix()})"
        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, md_text)


def save_blog(result: dict) -> Path | None:
    final_md = result.get("final") or result.get("md_with_placeholders") or result.get("merged_md") or ""
    plan = result.get("plan")
    title = get_value(plan, "blog_title", "generated_blog")
    if not final_md.strip():
        return None

    output_path = BLOGS_DIR / f"{safe_slug(title)}.md"
    output_path.write_text(final_md, encoding="utf-8")
    return output_path


def list_past_blogs() -> list[Path]:
    return sorted(BLOGS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)


def make_zip(markdown_text: str, title: str) -> bytes:
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{safe_slug(title)}.md", markdown_text)
        if IMAGES_DIR.exists():
            for image_path in IMAGES_DIR.rglob("*"):
                if image_path.is_file():
                    archive.write(image_path, f"images/{image_path.relative_to(IMAGES_DIR)}")
    memory_file.seek(0)
    return memory_file.getvalue()


def run_graph(topic: str, selected_date: date, progress_bar, status_box) -> dict:
    inputs = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": selected_date.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    current_state: dict = {}
    stage_labels = {
        "router": "Checking whether research is needed",
        "research": "Collecting relevant sources",
        "orchestrator": "Building the blog outline",
        "worker": "Writing blog sections",
        "merge_content": "Combining the sections",
        "decide_images": "Planning useful diagrams",
        "generate_and_place_images": "Generating and placing images",
    }

    try:
        for state in blog_graph.stream(inputs, stream_mode="values"):
            if isinstance(state, dict):
                current_state = state
                # Stream values show the latest state. The status message is updated
                # from fields as they become available; no second graph invocation.
                if current_state.get("plan"):
                    status_box.info("Outline created. Writing and assembling your blog…")
                if current_state.get("final"):
                    status_box.success("Your blog is ready!")
                progress_bar.progress(min(0.92, progress_bar.progress if False else 0.72))

        # A values stream normally yields the final complete state.
        if not current_state:
            raise RuntimeError("The graph did not return a state. Please check the backend output.")

        current_state.setdefault("final", "")
        if not current_state.get("final"):
            current_state["final"] = current_state.get("md_with_placeholders") or current_state.get("merged_md") or ""

        progress_bar.progress(1.0)
        status_box.success("Generation complete!")
        return current_state

    except Exception:
        raise


def render_plan(plan: Any) -> None:
    if not plan:
        st.info("The outline will appear here after blog generation.")
        return

    st.markdown(
        f"""
        <div class="soft-card">
            <div class="mini-label">YOUR BLOG OUTLINE</div>
            <div class="soft-card-title">{get_value(plan, "blog_title", "Untitled blog")}</div>
            <div class="soft-card-sub">
                Audience: {get_value(plan, "audience", "Not specified")}
                &nbsp; • &nbsp; Tone: {get_value(plan, "tone", "Not specified")}
                &nbsp; • &nbsp; Type: {get_value(plan, "blog_kind", "explainer")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    constraints = get_value(plan, "constraints", []) or []
    if constraints:
        st.markdown("**Writing constraints**")
        for item in constraints:
            st.markdown(f"- {item}")

    tasks = get_value(plan, "tasks", []) or []
    for index, task in enumerate(tasks, start=1):
        title = get_value(task, "title", f"Section {index}")
        goal = get_value(task, "goal", "")
        with st.expander(f"{index:02d}  ·  {title}", expanded=index == 1):
            if goal:
                st.write(goal)
            bullets = get_value(task, "bullets", []) or []
            for bullet in bullets:
                st.markdown(f"- {bullet}")
            c1, c2, c3 = st.columns(3)
            c1.caption(f"Target: {get_value(task, 'target_words', '—')} words")
            c2.caption("Code: " + ("Yes" if get_value(task, "requires_code", False) else "No"))
            c3.caption("Research: " + ("Yes" if get_value(task, "requires_research", False) else "No"))


def render_evidence(evidence: list) -> None:
    if not evidence:
        st.markdown(
            '<div class="empty-state"><strong>No research sources were needed or returned.</strong><br/>'
            'For evergreen topics, the agent may use its existing knowledge.</div>',
            unsafe_allow_html=True,
        )
        return

    rows = []
    for item in evidence:
        rows.append(
            {
                "Title": get_value(item, "title", ""),
                "Source": get_value(item, "source", "") or "",
                "Published": get_value(item, "published_at", "") or "Unknown",
                "URL": get_value(item, "url", ""),
                "Snippet": get_value(item, "snippet", "") or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for item in evidence:
        title = get_value(item, "title", "Source")
        url = get_value(item, "url", "")
        snippet = get_value(item, "snippet", "")
        if url:
            st.markdown(f"- [{title}]({url})")
            if snippet:
                st.caption(snippet)


def render_images(image_specs: list) -> None:
    if not image_specs:
        st.markdown(
            '<div class="empty-state"><strong>No diagrams were requested.</strong><br/>'
            'The agent only creates visuals when they add meaningful technical value.</div>',
            unsafe_allow_html=True,
        )
        return

    for spec in image_specs:
        spec = as_plain_dict(spec)
        filename = spec.get("filename", "")
        image_path = IMAGES_DIR / filename
        with st.container(border=True):
            st.markdown(f"**{spec.get('caption') or filename or 'Suggested visual'}**")
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
                st.caption(spec.get("alt", ""))
            else:
                st.warning("This image file was not found. The Markdown may contain an image-generation fallback.")
                if spec.get("prompt"):
                    with st.expander("View image prompt"):
                        st.write(spec["prompt"])


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 18px 0;">
            <div class="mini-label">YOUR WORKSPACE</div>
            <div style="font-family:'Space Grotesk';font-size:25px;font-weight:700;color:#F4F5FF;margin-top:7px;">
                BlogForge <span style="color:#9B8CFF;">✦</span>
            </div>
            <div style="font-size:13px;color:#9EABC5;margin-top:5px;">
                Your AI-powered technical writing studio.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Create a new blog")
    topic = st.text_area(
        "What should your blog be about?",
        placeholder="e.g. Self-Attention in Transformer Architecture",
        height=125,
        help="Enter a clear topic. The agent will plan, research when needed, and write the blog.",
    )
    selected_date = st.date_input("Reference date", value=date.today())
    generate_clicked = st.button(
        "✦  Generate my blog",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### Your saved blogs")
    past_blogs = list_past_blogs()
    if past_blogs:
        selected_blog = st.selectbox(
            "Open a previous blog",
            past_blogs,
            format_func=lambda path: path.stem.replace("_", " ").title(),
            label_visibility="collapsed",
        )
        if st.button("Open selected blog", use_container_width=True):
            st.session_state.loaded_blog = {
                "path": str(selected_blog),
                "content": selected_blog.read_text(encoding="utf-8"),
                "title": selected_blog.stem.replace("_", " ").title(),
            }
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("Your generated blogs will be saved here.")

    st.divider()
    st.caption("Built with Streamlit · LangGraph · Groq · Tavily · Gemini")


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-POWERED WRITING STUDIO</div>
        <h1 class="hero-title">Ideas into <span>impactful blogs.</span></h1>
        <div class="hero-subtitle">
            Research-backed technical writing, thoughtfully planned and generated by your multi-agent workflow.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Generate only when the user presses the button.
if generate_clicked:
    if not topic.strip():
        st.sidebar.error("Please enter a blog topic first.")
    else:
        st.session_state.loaded_blog = None
        st.session_state.logs = []
        st.session_state.last_topic = topic.strip()

        progress = st.progress(0)
        status = st.empty()
        status.info("Starting your writing agents…")

        try:
            result = run_graph(topic, selected_date, progress, status)
            st.session_state.result = result
            saved_path = save_blog(result)
            if saved_path:
                st.session_state.logs.append(f"Saved Markdown file: {saved_path.name}")
            st.rerun()
        except Exception as exc:
            status.error("The workflow stopped because an error occurred.")
            st.exception(exc)
            st.info(
                "If this is an API or model error, check NEW_API_KEY, TAVILY_API_KEY, "
                "GOOGLE_API_KEY and the backend terminal traceback."
            )


# ---------------------------------------------------------
# LOAD CURRENT OUTPUT
# ---------------------------------------------------------
result = st.session_state.result
loaded_blog = st.session_state.loaded_blog

if result:
    plan = result.get("plan")
    title = get_value(plan, "blog_title", "Generated blog")
    final_md = result.get("final") or result.get("md_with_placeholders") or result.get("merged_md") or ""
    evidence = result.get("evidence", []) or []
    image_specs = result.get("image_specs", []) or []
    mode = result.get("mode", "unknown")
    task_count = len(get_value(plan, "tasks", []) or []) if plan else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Blog sections", task_count)
    m2.metric("Research sources", len(evidence))
    m3.metric("Visuals planned", len(image_specs))
    m4.metric("Research mode", str(mode).replace("_", " ").title())

    st.markdown(
        f"""
        <div class="soft-card">
            <div class="mini-label">LATEST GENERATION</div>
            <div class="soft-card-title">{title}</div>
            <div class="soft-card-sub">Topic: {st.session_state.last_topic}</div>
            <div style="margin-top:12px;"><span class="status-pill">✓ Generation complete</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_plan, tab_research, tab_preview, tab_images, tab_activity = st.tabs(
        ["✦  Blog plan", "⌕  Research", "✍  Blog preview", "▧  Visuals", "☷  Activity"]
    )

    with tab_plan:
        render_plan(plan)

    with tab_research:
        st.markdown("### Research library")
        st.caption("Sources collected by the research agent for this blog.")
        render_evidence(evidence)

    with tab_preview:
        st.markdown("### Your generated article")
        st.caption("Review the Markdown output below. You can download it or bundle it with generated images.")
        if final_md.strip():
            st.markdown(markdown_image_path(final_md))
            st.divider()
            left, right = st.columns(2)
            left.download_button(
                "↓  Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=f"{safe_slug(title)}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            right.download_button(
                "↓  Download blog + images (.zip)",
                data=make_zip(final_md, title),
                file_name=f"{safe_slug(title)}_project.zip",
                mime="application/zip",
                use_container_width=True,
            )
            with st.expander("View raw Markdown"):
                st.code(final_md, language="markdown")
        else:
            st.warning("The graph completed, but no final Markdown was returned.")

    with tab_images:
        st.markdown("### Generated visuals")
        st.caption("Technical diagrams selected by the image-planning step.")
        render_images(image_specs)

    with tab_activity:
        st.markdown("### Workflow activity")
        st.markdown(
            """
            <div class="soft-card">
                <div class="soft-card-title">Agent workflow</div>
                <div class="soft-card-sub">Router → Research (if needed) → Outline → Parallel section writers → Merge → Image planning → Final blog</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        stages = [
            ("01", "Router", "Decides whether the topic needs fresh web research."),
            ("02", "Research", "Collects and filters relevant evidence when required."),
            ("03", "Orchestrator", "Creates the blog title, audience, tone and section plan."),
            ("04", "Section writers", "Writes the planned sections in parallel."),
            ("05", "Reducer", "Merges sections and plans useful technical diagrams."),
            ("06", "Image generation", "Generates images and inserts them into the Markdown."),
        ]
        for number, name, description in stages:
            st.markdown(
                f"""
                <div class="soft-card" style="padding:13px 17px;">
                    <span style="color:#9B8CFF;font-weight:800;margin-right:12px;">{number}</span>
                    <span style="color:#F2F5FF;font-weight:700;">{name}</span>
                    <div class="soft-card-sub" style="margin:5px 0 0 30px;">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif loaded_blog:
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="mini-label">SAVED ARTICLE</div>
            <div class="soft-card-title">{loaded_blog['title']}</div>
            <div class="soft-card-sub">Opened from your generated blogs folder.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(markdown_image_path(loaded_blog["content"]))
    st.download_button(
        "↓  Download this Markdown",
        data=loaded_blog["content"].encode("utf-8"),
        file_name=f"{safe_slug(loaded_blog['title'])}.md",
        mime="text/markdown",
    )

else:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:34px;margin-bottom:10px;">✦</div>
            <strong>Your next great technical blog starts here.</strong>
            <div style="margin-top:8px;font-size:13px;">
                Enter a topic in the left workspace and click Generate my blog.
                Your outline, research, article and visuals will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What your writing studio does")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="soft-card">
                <div style="font-size:24px;">⌕</div>
                <div class="soft-card-title">Smart research</div>
                <div class="soft-card-sub">Finds current sources when the topic needs fresh information.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="soft-card">
                <div style="font-size:24px;">◈</div>
                <div class="soft-card-title">Structured writing</div>
                <div class="soft-card-sub">Creates a detailed outline and writes sections through parallel agents.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="soft-card">
                <div style="font-size:24px;">▧</div>
                <div class="soft-card-title">Useful visuals</div>
                <div class="soft-card-sub">Plans and generates technical diagrams when they improve understanding.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div style="text-align:center;color:#64738F;font-size:11px;padding:35px 0 12px 0;">
        BLOGFORGE · AI TECHNICAL WRITING STUDIO
    </div>
    """,
    unsafe_allow_html=True,
)
