"""
Smart Campus Navigator — BSBI
Sidebar layout + unified input bar with all 3 modalities.
"""
import os, sys, io, html as _html
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="BSBI Campus Navigator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#EEF3F8;color:#111827;}

/* Sidebar */
[data-testid="stSidebar"]{
  background:#FFFFFF !important;
  border-right:1px solid #E5E9F0 !important;
  box-shadow:2px 0 8px rgba(0,0,0,0.04);
  position:fixed !important;
  height:100vh !important;
  overflow:hidden !important;
}
[data-testid="stSidebar"] .block-container{
  padding:1rem !important;
  height:100vh !important;
  overflow-y:auto !important;
  scrollbar-width:thin;
}
[data-testid="stSidebar"] .block-container::-webkit-scrollbar{width:6px;}
[data-testid="stSidebar"] .block-container::-webkit-scrollbar-thumb{
  background:#CBD5E1;border-radius:999px;
}
#MainMenu,footer,header{visibility:hidden;}

/* Main area - no extra padding */
.main .block-container{
  padding:0.5rem 1.5rem 0 1.5rem !important;
  max-width:900px;
}

/* Header */
.top-bar{
  display:flex;align-items:center;gap:0.75rem;
  padding:0.4rem 0 0.5rem 0;
  border-bottom:1px solid #E5E9F0;
  margin-bottom:0.5rem;
}
.top-logo{
  width:36px;height:36px;
  background:linear-gradient(135deg,#2563EB,#1D4ED8);
  border-radius:9px;display:flex;align-items:center;
  justify-content:center;font-size:1.1rem;
  box-shadow:0 2px 6px rgba(37,99,235,0.25);
}
.top-title{font-size:1rem;font-weight:700;color:#111827;margin:0;}
.top-sub{font-size:0.65rem;color:#1D4ED8;font-family:'JetBrains Mono',monospace;
         text-transform:uppercase;letter-spacing:0.08em;margin:0;}
.top-live{margin-left:auto;display:flex;align-items:center;gap:0.35rem;
          font-size:0.62rem;color:#6B7280;font-family:'JetBrains Mono',monospace;}
.ldot{width:7px;height:7px;border-radius:50%;background:#10B981;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}

/* Messages */
.msg-wrap{display:flex;flex-direction:column;gap:0.1rem;margin-bottom:0.6rem;}
.msg-user{display:flex;justify-content:flex-end;}
.msg-bot{display:flex;justify-content:flex-start;align-items:flex-start;gap:0.5rem;}
.bot-av{width:26px;height:26px;border-radius:7px;flex-shrink:0;margin-top:2px;
        background:linear-gradient(135deg,#2563EB,#1D4ED8);
        display:flex;align-items:center;justify-content:center;font-size:0.8rem;}
.bub-u{background:#1D4ED8;color:#FFFFFF;border-radius:18px 18px 4px 18px;
       padding:0.65rem 1rem;max-width:70%;font-size:0.88rem;line-height:1.6;
       word-wrap:break-word;box-shadow:0 3px 10px rgba(29,78,216,0.22);}
.bub-b{background:#FFFFFF;color:#1F2937;border-radius:4px 18px 18px 18px;
       padding:0.65rem 1rem;max-width:82%;font-size:0.88rem;line-height:1.65;
       word-wrap:break-word;border:1px solid #D8E1EC;
       box-shadow:0 2px 8px rgba(15,23,42,0.08);}
.msg-meta{font-size:0.58rem;color:#64748B;font-family:'JetBrains Mono',monospace;
          margin-top:0.18rem;padding:0 0.2rem;}
.msg-meta-r{text-align:right;}

/* Badges */
.badge{display:inline-block;font-size:0.56rem;font-weight:500;
       font-family:'JetBrains Mono',monospace;padding:0.08rem 0.38rem;
       border-radius:20px;border:1px solid;text-transform:uppercase;
       letter-spacing:0.04em;margin-left:0.25rem;vertical-align:middle;}
.b-text{color:#1D4ED8;border-color:#93C5FD;background:#DBEAFE;}
.b-voice{color:#047857;border-color:#6EE7B7;background:#D1FAE5;}
.b-image{color:#6D28D9;border-color:#C4B5FD;background:#EDE9FE;}
.b-fusion{color:#B45309;border-color:#FDBA74;background:#FFEDD5;}
.b-hi{color:#047857;border-color:#6EE7B7;background:#D1FAE5;}
.b-lo{color:#B91C1C;border-color:#FCA5A5;background:#FEE2E2;}

/* KB card */
.kb-card{margin-top:0.6rem;background:#F8FAFC;border-radius:10px;
         padding:0.65rem 0.85rem;border:1px solid #CBD5E1;border-left:4px solid #1D4ED8;}
.kb-name{font-weight:700;color:#1E3A8A;margin-bottom:0.3rem;font-size:0.82rem;}
.kb-row{display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.2rem;}
.kb-item{font-size:0.67rem;color:#475569;font-family:'JetBrains Mono',monospace;}
.kb-item span{color:#1F2937;font-family:'Inter',sans-serif;}
.kb-events{margin-top:0.35rem;color:#92400E;font-size:0.73rem;}
.kb-ev-lbl{font-family:'JetBrains Mono',monospace;font-size:0.58rem;
           text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:0.1rem;}

/* Empty state */
.empty-state{text-align:center;padding:2rem 1rem;}
.empty-icon{font-size:2.2rem;margin-bottom:0.5rem;}
.empty-title{font-size:1rem;font-weight:600;color:#1A1D23;margin-bottom:0.3rem;}
.empty-sub{font-size:0.82rem;color:#6B7280;line-height:1.7;}

.composer-status{
  font-size:0.64rem;color:#1D4ED8;
  font-family:'JetBrains Mono',monospace;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  padding:0 0.25rem 0.3rem 0.25rem;
}
.composer-hint{
  font-size:0.62rem;color:#64748B;
  font-family:'JetBrains Mono',monospace;
  padding:0 0.25rem 0.3rem 0.25rem;
}
.input-label{font-size:0.6rem;font-family:'JetBrains Mono',monospace;
             color:#9CA3AF;text-transform:uppercase;letter-spacing:0.08em;
             margin-bottom:0.3rem;}
.modality-row{display:flex;gap:0.4rem;margin-top:0.5rem;
              padding-top:0.5rem;border-top:1px solid #F1F5F9;}
.mod-active{font-size:0.72rem;color:#2563EB;font-weight:500;
            padding:0.15rem 0.5rem;border-radius:6px;
            background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.2);}

/* Attachment preview */
.attach-box{background:#DBEAFE;border:1px solid #93C5FD;border-radius:8px;
            padding:0.35rem 0.7rem;font-size:0.75rem;color:#1E40AF;
            margin:0.25rem 0 0.4rem 0;display:flex;align-items:center;gap:0.4rem;}

/* Transcript */
.tr-box{background:#F0FDF4;border:1.5px dashed #86EFAC;border-radius:8px;
        padding:0.45rem 0.75rem;font-size:0.78rem;
        font-family:'JetBrains Mono',monospace;color:#059669;
        margin-bottom:0.4rem;line-height:1.5;}

/* Sidebar elements */
.sb-head{display:flex;align-items:center;gap:0.55rem;
         padding-bottom:0.7rem;border-bottom:1px solid #F0F2F5;margin-bottom:0.5rem;}
.sb-logo{width:30px;height:30px;background:linear-gradient(135deg,#2563EB,#1D4ED8);
         border-radius:8px;display:flex;align-items:center;justify-content:center;
         font-size:0.95rem;box-shadow:0 2px 4px rgba(37,99,235,0.25);}
.sb-title{font-size:0.85rem;font-weight:600;color:#1A1D23;}
.sb-sub{font-size:0.6rem;color:#1D4ED8;font-family:'JetBrains Mono',monospace;
        text-transform:uppercase;letter-spacing:0.08em;}
.sb-sec{font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#1D4ED8;
        text-transform:uppercase;letter-spacing:0.1em;margin:0.65rem 0 0.35rem 0;}
.sb-div{border-top:1px solid #F0F2F5;margin:0.65rem 0;}

/* Buttons */
.stButton>button{
  width:100%;background:linear-gradient(135deg,#1D4ED8,#1E40AF) !important;
  color:#fff !important;border:none !important;border-radius:9px !important;
  padding:0.5rem !important;font-weight:600 !important;font-size:0.82rem !important;
  box-shadow:0 2px 5px rgba(37,99,235,0.25) !important;transition:all 0.15s !important;
}
.stButton>button:hover{opacity:0.9 !important;}
.sug .stButton>button{
  background:#FFFFFF !important;color:#334155 !important;
  border:1px solid #CBD5E1 !important;font-weight:500 !important;
  font-size:0.76rem !important;box-shadow:none !important;
}
.sug .stButton>button:hover{border-color:#1D4ED8 !important;color:#1D4ED8 !important;}
.sec .stButton>button{
  background:rgba(37,99,235,0.07) !important;color:#2563EB !important;
  border:1px solid rgba(37,99,235,0.2) !important;
  font-weight:500 !important;box-shadow:none !important;
}
.clr .stButton>button{
  background:transparent !important;border:1px solid #FEE2E2 !important;
  color:#EF4444 !important;font-weight:400 !important;
  font-size:0.72rem !important;box-shadow:none !important;
}

/* Streamlit overrides */
.stTextArea textarea{
  background:transparent !important;border:none !important;
  color:#1A1D23 !important;font-size:0.9rem !important;
  resize:none !important;box-shadow:none !important;padding:0 !important;
}
.stTextArea textarea:focus{box-shadow:none !important;border:none !important;}
.stTextInput input{
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  color:#1A1D23 !important;
  font-size:0.92rem !important;
  padding-top:0.35rem !important;
}
.stTextInput input:focus{
  border:none !important;
  box-shadow:none !important;
}
div[data-testid="stForm"]{
  background:#FFFFFF !important;
  border:1.5px solid #C7D2E1 !important;
  border-radius:18px !important;
  padding:0.45rem 0.65rem 0.15rem 0.65rem !important;
  margin:0.45rem 0 0.35rem 0 !important;
  box-shadow:0 2px 8px rgba(0,0,0,0.05) !important;
}
div[data-testid="stForm"]:focus-within{
  border-color:#1D4ED8 !important;
  box-shadow:0 2px 10px rgba(37,99,235,0.12) !important;
}
div[data-testid="stForm"] .stTextInput input{
  color:#1A1D23 !important;
  font-size:0.92rem !important;
}
[data-testid="stCameraInput"]>div{
  border:1.5px dashed #CBD5E1 !important;border-radius:10px !important;
  background:#F8FAFC !important;
}
.stFileUploader>div{
  border:1.5px dashed #CBD5E1 !important;border-radius:8px !important;
  background:#F8FAFC !important;
}
[data-testid="stCameraInput"] label,.stTextArea label,
.stFileUploader label,.stTextInput label{display:none !important;}
.stTabs [data-baseweb="tab-list"]{
  background:#F1F5F9;border-radius:7px;padding:2px;gap:2px;border:1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"]{background:transparent;color:#6B7280;
  border-radius:5px;font-size:0.78rem;padding:0.3rem 0.6rem;}
.stTabs [aria-selected="true"]{background:#FFFFFF !important;color:#2563EB !important;
  box-shadow:0 1px 2px rgba(0,0,0,0.07) !important;}
.audio-row{display:flex;justify-content:center;padding:0.4rem 0;}
.icon-btn .stButton>button,
.icon-btn .stFormSubmitButton>button{
  background:#F8FAFC !important;color:#334155 !important;
  border:1.5px solid #C7D2E1 !important;border-radius:50% !important;
  padding:0 !important;font-size:1.05rem !important;
  font-weight:400 !important;box-shadow:none !important;
  height:38px !important;min-height:38px !important;
}
.icon-btn .stButton>button:hover,
.icon-btn .stFormSubmitButton>button:hover{
  border-color:#1D4ED8 !important;color:#1D4ED8 !important;
  background:#DBEAFE !important;
}
.icon-btn-active .stButton>button,
.icon-btn-active .stFormSubmitButton>button{
  background:#DBEAFE !important;color:#1D4ED8 !important;
  border:1.5px solid #1D4ED8 !important;border-radius:50% !important;
  padding:0 !important;font-size:1.05rem !important;
  font-weight:400 !important;box-shadow:none !important;
  height:38px !important;min-height:38px !important;
}
.icon-btn-danger .stButton>button,
.icon-btn-danger .stFormSubmitButton>button{
  background:#FEE2E2 !important;color:#B91C1C !important;
  border:1.5px solid #FCA5A5 !important;border-radius:50% !important;
  padding:0 !important;font-size:1rem !important;
  font-weight:400 !important;box-shadow:none !important;
  height:38px !important;min-height:38px !important;
}
.send-btn .stButton>button,
.send-btn .stFormSubmitButton>button{
  background:#111827 !important;color:#FFFFFF !important;
  border:none !important;border-radius:50% !important;
  padding:0 !important;font-size:1rem !important;
  font-weight:700 !important;box-shadow:none !important;
  height:38px !important;min-height:38px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "show_voice"   not in st.session_state: st.session_state.show_voice   = False
if "show_image"   not in st.session_state: st.session_state.show_image   = False
if "pending_img"  not in st.session_state: st.session_state.pending_img  = None
if "pending_name" not in st.session_state: st.session_state.pending_name = ""
if "audio_data"   not in st.session_state: st.session_state.audio_data   = None
if "audio_suffix" not in st.session_state: st.session_state.audio_suffix = ".wav"
if "voice_hint"   not in st.session_state: st.session_state.voice_hint   = ""

# ── Pipelines ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipelines():
    from src.text_pipeline  import run_text_pipeline
    from src.voice_pipeline import run_voice_pipeline_from_bytes
    return run_text_pipeline, run_voice_pipeline_from_bytes

# ── Helpers ───────────────────────────────────────────────────────────
def mbadge(m):
    B={"text":'<span class="badge b-text">⌨ text</span>',
       "voice":'<span class="badge b-voice">🎤 voice</span>',
       "image":'<span class="badge b-image">📷 image</span>',
       "text+image":'<span class="badge b-fusion">⚡ fusion</span>',
       "image+text":'<span class="badge b-fusion">⚡ fusion</span>',
       "text+voice":'<span class="badge b-fusion">⌨🎤 fusion</span>',
       "voice+image":'<span class="badge b-fusion">🎤📷 fusion</span>',
       "text+voice+image":'<span class="badge b-fusion">⌨🎤📷 fusion</span>'}
    return B.get(m,B["text"])

def kb_card(result):
    loc=result.get("matched_location",""); conf=result.get("intent_confidence",0)
    kb=result.get("kb_result","")
    if not loc or loc in ("fallback","greeting","no_match","error",""): return ""
    hours=coord=""; events=[]; related=[]; section=""
    for line in kb.splitlines():
        ll=line.lower().strip()
        if ll.startswith("events:"):
            section="events"; continue
        if ll.startswith("additional related locations:"):
            section="related"; continue
        if ll.startswith("courses offered:"):
            section="courses"; continue
        if (ll.startswith("opening_hours") or ll.startswith("hours")) and not hours:
            v=line.split(":",1)[-1].strip()
            if v and v.lower() not in ("","location","description"): hours=v
        if (ll.startswith("zone") or ll.startswith("location:")) and not coord:
            v=line.split(":",1)[-1].strip()
            if v and len(v)<80: coord=v
        if section=="events" and ll.startswith("- ") and len(line.strip())>5:
            events.append(_html.escape(line.strip().lstrip("- ")))
        if section=="related" and ll.startswith("- ") and len(line.strip())>5:
            related.append(_html.escape(line.strip().lstrip("- ")))
    cb="b-hi" if conf>0.7 else "b-lo"
    rows="".join([
        f"<div class='kb-item'>HOURS <span>{_html.escape(hours)}</span></div>" if hours else "",
        f"<div class='kb-item'>ZONE  <span>{_html.escape(coord)}</span></div>" if coord else "",
        f"<div class='kb-item'>CONF  <span class='badge {cb}'>{conf*100:.0f}%</span></div>",
    ])
    ev=("".join(["<div class='kb-events'><div class='kb-ev-lbl'>Events</div>",
                 *[f"<div>🗓 {e}</div>" for e in events[:3]],"</div>"])) if events else ""
    rel=("".join(["<div class='kb-events'><div class='kb-ev-lbl'>Related locations</div>",
                  *[f"<div>• {r}</div>" for r in related[:2]],"</div>"])) if related else ""
    return (f"<div class='kb-card'><div class='kb-name'>📍 {_html.escape(loc)}</div>"
            f"<div class='kb-row'>{rows}</div>{ev}{rel}</div>")

def add_msg(role,content,modality="text",result=None):
    st.session_state.messages.append({
        "role":role,"content":content,"modality":modality,
        "result":result,"time":datetime.now().strftime("%H:%M")})

def do_text(q, mod="text"):
    run_text,_=load_pipelines()
    add_msg("user",q,mod)
    r=run_text(q,modality=mod)
    add_msg("assistant",r["response"],mod,r)

def do_image(img_bytes, hint="", img_name="photo"):
    from src.image_pipeline  import run_image_pipeline
    from src.answer_generator import build_kb_result
    from src.llm_handler      import ask_llm
    pil=Image.open(io.BytesIO(img_bytes)).convert("RGB")
    ir=run_image_pipeline(pil)
    top=ir.get("top_match"); conf=ir.get("confidence",0)
    low=ir.get("low_confidence",True)
    mod="text+image" if hint.strip() else "image"
    disp=f"📷 {img_name}"+(f' + "{hint.strip()}"' if hint.strip() else "")
    if low and not hint.strip():
        top3=[r["record"]["name"] for r in ir.get("top_k",[])]
        msg=(f"I'm not confident (score: {conf:.2f}). "
             f"Best guesses: {', '.join(top3[:3])}. "
             "Try adding a text hint or a clearer photo.")
        add_msg("user",disp,"image"); add_msg("assistant",msg,"image",{})
    elif top:
        q_llm=hint.strip() or f"Tell me about {top['name']}"
        kb_str=build_kb_result(q_llm,"facility_info",top)
        resp,_=ask_llm(q_llm,kb_str)
        pr={"response":resp,"intent":"facility_info","intent_confidence":conf,
            "matched_location":top["name"],"retrieval_score":conf,
            "llm_used":True,"kb_result":kb_str}
        add_msg("user",disp,mod); add_msg("assistant",resp,mod,pr)
    else:
        add_msg("user",disp,"image")
        add_msg("assistant","Couldn't identify a campus location. "
                "Try a clearer photo or add a text hint.","image",{})

def do_multimodal_fusion(img_bytes=None, audio_bytes=None, audio_suffix=".wav",
                         text_hint="", img_name="photo"):
    from src.answer_generator import build_kb_result
    from src.fusion_layer     import run_fusion_pipeline
    from src.llm_handler      import ask_llm

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB") if img_bytes else None
    fusion = run_fusion_pipeline(
        text_query=text_hint.strip() or None,
        image_input=pil,
        voice_input=audio_bytes,
        voice_suffix=audio_suffix,
        top_k=3,
    )

    if fusion.get("error") and not fusion.get("top_match"):
        if img_bytes:
            do_image(img_bytes, hint=text_hint, img_name=img_name)
        elif audio_bytes:
            _, run_voice = load_pipelines()
            r = run_voice(audio_bytes, suffix=audio_suffix)
            transcript = r.get("transcript", "")
            if text_hint.strip():
                do_text(f"{text_hint.strip()} {transcript}".strip(), "text+voice")
            else:
                add_msg("user", f'🎤 "{transcript}"' if transcript else "🎤 Voice query", "voice")
                add_msg("assistant", r["response"], "voice", r)
        return

    top = fusion.get("top_match")
    transcript = fusion.get("transcript", "").strip()
    parts = []
    if img_bytes:
        parts.append(f"📷 {img_name}")
    if transcript:
        parts.append(f'🎤 "{transcript}"')
    if text_hint.strip():
        parts.append(f'"{text_hint.strip()}"')
    disp = " + ".join(parts) if parts else "Multimodal query"

    used = set(fusion.get("modalities_used", []))
    if {"text", "voice", "image"}.issubset(used):
        mod = "text+voice+image"
    elif {"text", "voice"}.issubset(used):
        mod = "text+voice"
    elif {"voice", "image"}.issubset(used):
        mod = "voice+image"
    elif {"text", "image"}.issubset(used):
        mod = "text+image"
    elif "image" in used:
        mod = "image"
    elif "voice" in used:
        mod = "voice"
    else:
        mod = "text"

    if not top:
        add_msg("user", disp, mod)
        add_msg("assistant",
                "I could not confidently match that to a campus location. "
                "Try a clearer image, a shorter voice query, or a typed hint.",
                mod, {})
        return

    query_text = " ".join([text_hint.strip(), transcript]).strip()
    if not query_text:
        query_text = f"Tell me about {top.get('name', 'this place')}"

    score = float((fusion.get("top_k_records") or [{}])[0].get("score", 0.0))
    kb_str = build_kb_result(query_text, "facility_info", top)
    resp, used_llm = ask_llm(query_text, kb_str)
    pr = {
        "response": resp,
        "intent": "multimodal_fusion",
        "intent_confidence": score,
        "matched_location": top.get("name", ""),
        "retrieval_score": score,
        "llm_used": used_llm,
        "kb_result": kb_str,
        "transcript": transcript,
        "modalities_used": fusion.get("modalities_used", []),
    }
    add_msg("user", disp, mod)
    add_msg("assistant", resp, mod, pr)

def render_chat():
    if not st.session_state.messages:
        st.markdown("""<div class='empty-state'>
          <div class='empty-icon'>🏛️</div>
          <div class='empty-title'>BSBI Campus Navigator</div>
          <div class='empty-sub'>Ask about locations, hours, events, admissions,<br>
          clubs, scholarships, or upload a campus photo.</div>
        </div>""", unsafe_allow_html=True)
        return
    for msg in st.session_state.messages:
        sc=_html.escape(msg["content"]); t=msg["time"]; m=msg["modality"]
        if msg["role"]=="user":
            st.markdown(
                f'<div class="msg-wrap">'
                f'<div class="msg-user"><div class="bub-u">{sc}</div></div>'
                f'<div class="msg-meta msg-meta-r">{t} {mbadge(m)}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            r=msg.get("result") or {}
            intent=r.get("intent",""); conf=r.get("intent_confidence",0)
            cb="b-hi" if conf>0.7 else "b-lo"
            itag=(f'<span class="badge {cb}">{_html.escape(intent.replace("_"," "))}</span>'
                  if intent and intent not in ("greeting","fallback","error","") else "")
            # Render newlines as <br> for proper formatting
            sc_html = sc.replace("\n", "<br>")
            st.markdown(
                f'<div class="msg-wrap">'
                f'<div class="msg-bot">'
                f'<div class="bot-av">🤖</div>'
                f'<div class="bub-b">{sc_html}{kb_card(r)}</div>'
                f'</div>'
                f'<div class="msg-meta">Campus AI {mbadge(m)}{itag} · {t}</div>'
                f'</div>', unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class='sb-head'>
      <div class='sb-logo'>🏛️</div>
      <div><div class='sb-title'>Campus Navigator</div>
           <div class='sb-sub'>BSBI Berlin</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sb-sec'>How to use</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.78rem;color:#6B7280;line-height:1.8;'>
    ⌨️ <b>Text</b> — type any question<br>
    🎤 <b>Voice</b> — click mic, speak, send<br>
    📷 <b>Image</b> — camera or upload<br>
    ⚡ <b>Combine</b> — image + text, voice + text, or all three
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-sec'>Quick questions</div>", unsafe_allow_html=True)

    SUGS = [
        "Where is the library?",
        "Cafeteria opening hours",
        "Price of coffee",
        "How to apply for scholarship?",
        "How to join a club?",
        "Any events this week?",
        "I lost my student card",
        "Where is the gym?",
        "How to get to placement cell?",
        "Hostel fees",
    ]
    for s in SUGS:
        st.markdown("<div class='sug'>", unsafe_allow_html=True)
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            do_text(s); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-div'></div>", unsafe_allow_html=True)
    if st.session_state.messages:
        st.markdown("<div class='clr'>", unsafe_allow_html=True)
        if st.button("🗑 Clear conversation", key="clr", use_container_width=True):
            st.session_state.messages=[]
            st.session_state.pending_img=None
            st.session_state.audio_data=None
            st.session_state.audio_suffix=".wav"
            st.session_state.show_voice=False
            st.session_state.show_image=False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────
st.markdown("""<div class='top-bar'>
  <div class='top-logo'>🏛️</div>
  <div>
    <p class='top-title'>Campus Navigator</p>
    <p class='top-sub'>Berlin School of Business &amp; Innovation · AI Assistant</p>
  </div>
  <div class='top-live'><div class='ldot'></div>Live</div>
</div>""", unsafe_allow_html=True)

# Chat messages
render_chat()

# ── Voice/Image panels (open above composer) ──────────────────────────
if st.session_state.show_voice:
    with st.container():
        st.markdown("""<div style='background:#F0FDF4;border:1px solid #BBF7D0;
          border-radius:12px;padding:0.9rem 1rem;margin-top:0.5rem;'>
          <div style='font-size:0.65rem;color:#059669;font-family:JetBrains Mono,monospace;
          text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>
          🎤 Voice Input — record or upload audio</div>
          <div style='font-size:0.76rem;color:#047857;line-height:1.5;margin-bottom:0.6rem;'>
          If the mic does not capture, allow microphone permission for localhost in the browser,
          or upload a short WAV/MP3/M4A file below.
          </div>""",
                    unsafe_allow_html=True)
        try:
            from audio_recorder_streamlit import audio_recorder
            col_rec, col_send = st.columns([1,2])
            with col_rec:
                st.markdown("<div style='display:flex;justify-content:center;'>",
                            unsafe_allow_html=True)
                audio_bytes = audio_recorder(
                    text="", recording_color="#059669",
                    neutral_color="#9CA3AF", icon_size="2x")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_send:
                voice_hint = st.text_input(
                    "voice_hint_input",
                    placeholder="Optional typed context (combines with voice/image)",
                    key="voice_hint",
                    label_visibility="collapsed",
                )
                uploaded_audio = st.file_uploader(
                    "audio_upload",
                    type=["wav", "mp3", "m4a", "ogg", "webm"],
                    key="voice_upload",
                    label_visibility="collapsed",
                )
                if audio_bytes:
                    st.session_state.audio_data = audio_bytes
                    st.session_state.audio_suffix = ".wav"
                    st.markdown(
                        "<div class='tr-box'>✓ Audio captured — press Send</div>",
                        unsafe_allow_html=True)
                if uploaded_audio:
                    st.session_state.audio_data = uploaded_audio.getvalue()
                    suffix = "." + uploaded_audio.name.split(".")[-1].lower()
                    st.session_state.audio_suffix = suffix
                    st.markdown(
                        f"<div class='tr-box'>✓ Uploaded {_html.escape(uploaded_audio.name)} — press Send</div>",
                        unsafe_allow_html=True)
                if st.session_state.audio_data:
                    if st.button("Send Voice →", key="send_v", use_container_width=True):
                        with st.spinner("Transcribing…"):
                            suffix = st.session_state.get("audio_suffix", ".wav")
                            hint_text = voice_hint.strip()
                            if st.session_state.pending_img:
                                do_multimodal_fusion(
                                    img_bytes=st.session_state.pending_img,
                                    audio_bytes=st.session_state.audio_data,
                                    audio_suffix=suffix,
                                    text_hint=hint_text,
                                    img_name=st.session_state.pending_name,
                                )
                                st.session_state.pending_img=None
                                st.session_state.pending_name=""
                            elif hint_text:
                                _, run_voice = load_pipelines()
                                r = run_voice(st.session_state.audio_data, suffix=suffix)
                                tr = r.get("transcript", "")
                                combined = f"{hint_text} {tr}".strip()
                                do_text(combined, "text+voice")
                            else:
                                _,run_voice=load_pipelines()
                                r=run_voice(st.session_state.audio_data, suffix=suffix)
                                tr=r.get("transcript","")
                                disp=f'🎤 "{tr}"' if tr else "🎤 Voice query"
                                add_msg("user",disp,"voice")
                                add_msg("assistant",r["response"],"voice",r)
                            st.session_state.audio_data=None
                            st.session_state.audio_suffix=".wav"
                            st.session_state.show_voice=False
                        st.rerun()
        except ImportError:
            st.error("Run: pip install audio-recorder-streamlit")
        st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.show_image:
    with st.container():
        st.markdown("""<div style='background:#F5F3FF;border:1px solid #DDD6FE;
          border-radius:12px;padding:0.9rem 1rem;margin-top:0.5rem;'>
          <div style='font-size:0.65rem;color:#7C3AED;font-family:JetBrains Mono,monospace;
          text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>
          📷 Image Input — camera or upload</div>""",
                    unsafe_allow_html=True)

        t1, t2 = st.tabs(["📁 Upload", "📷 Camera"])
        img_data = None
        with t1:
            upl = st.file_uploader("upl", type=["jpg","jpeg","png","webp"],
                                   key="upl2", label_visibility="collapsed")
            if upl: img_data = upl
        with t2:
            cam = st.camera_input("cam", key="cam2",
                                  label_visibility="collapsed")
            if cam: img_data = cam

        hint = st.text_input("hint2",
                             placeholder="Optional text hint (e.g. What is this building?)",
                             key="img_hint2", label_visibility="collapsed")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("📷 Identify & Ask", key="send_img2",
                         use_container_width=True):
                if img_data:
                    with st.spinner(""):
                        do_image(img_data.getvalue(),
                                 hint=hint.strip(),
                                 img_name=getattr(img_data,"name","photo"))
                    st.session_state.show_image=False
                    st.rerun()
                else:
                    st.warning("Capture or upload first.")
        with b2:
            if img_data:
                if st.button("📎 Attach to message", key="attach2",
                             use_container_width=True):
                    st.session_state.pending_img  = img_data.getvalue()
                    st.session_state.pending_name = getattr(img_data,"name","photo")
                    st.session_state.show_image   = False
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── Composer ──────────────────────────────────────────────────────────
active_modes = []
if st.session_state.show_voice:
    active_modes.append("voice")
if st.session_state.show_image:
    active_modes.append("image")
if st.session_state.pending_img:
    active_modes.append("image attached")

# Attachment preview
if st.session_state.pending_img:
    st.markdown(
        f"<div class='attach-box'>📷 <b>{_html.escape(st.session_state.pending_name)}</b> attached "
        f"<span style='color:#6B7280;font-size:0.68rem'>— send with text, voice, or both</span></div>",
        unsafe_allow_html=True)

with st.form("campus_composer", clear_on_submit=True):
    c_plus, c_text, c_mic, c_send = st.columns([0.45, 5.2, 0.45, 0.45])
    with c_plus:
        cls = "icon-btn-danger" if st.session_state.pending_img else ("icon-btn-active" if st.session_state.show_image else "icon-btn")
        st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
        plus_label = "×" if st.session_state.pending_img else "📎"
        plus_pressed = st.form_submit_button(
            plus_label,
            help="Attach or remove image",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_text:
        q = st.text_input(
            "composer_text",
            placeholder="Ask anything about the campus...",
            key="composer_text",
            label_visibility="collapsed",
        )
    with c_mic:
        cls = "icon-btn-active" if st.session_state.show_voice else "icon-btn"
        st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
        mic_pressed = st.form_submit_button(
            "🎤",
            help="Record or upload voice",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_send:
        st.markdown("<div class='send-btn'>", unsafe_allow_html=True)
        send_pressed = st.form_submit_button(
            "↑",
            help="Send message",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

if active_modes:
    st.markdown(
        f"<div class='composer-status'>{_html.escape(' + '.join(active_modes))}</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='composer-hint'>text · voice · image</div>", unsafe_allow_html=True)

submit_text = q.strip()

if (send_pressed or (plus_pressed and submit_text)) and submit_text:
    if st.session_state.pending_img:
        with st.spinner(""):
            do_image(st.session_state.pending_img, hint=submit_text,
                     img_name=st.session_state.pending_name)
        st.session_state.pending_img  = None
        st.session_state.pending_name = ""
    else:
        do_text(submit_text, "text")
    st.rerun()

if plus_pressed:
    if st.session_state.pending_img:
        st.session_state.pending_img  = None
        st.session_state.pending_name = ""
    else:
        st.session_state.show_image = not st.session_state.show_image
        st.session_state.show_voice = False
    st.rerun()

if mic_pressed:
    st.session_state.show_voice = not st.session_state.show_voice
    st.session_state.show_image = False
    st.rerun()

components.html(
    """
    <script>
    const doc = window.parent.document;
    const focusComposer = () => {
      const input = doc.querySelector('input[aria-label="composer_text"], input[placeholder="Ask anything about the campus..."]');
      if (input && doc.activeElement !== input) {
        input.focus({ preventScroll: true });
      }
    };
    const scrollMainToBottom = () => {
      const main = doc.querySelector('section.main');
      const target = doc.querySelector('div[data-testid="stForm"]');
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "end" });
      } else if (main) {
        main.scrollTo({ top: main.scrollHeight, behavior: "smooth" });
      } else {
        window.parent.scrollTo({ top: doc.body.scrollHeight, behavior: "smooth" });
      }
      setTimeout(focusComposer, 150);
    };
    setTimeout(scrollMainToBottom, 120);
    </script>
    """,
    height=0,
)
