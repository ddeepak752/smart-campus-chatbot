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
.stApp{background:#F4F7FB;color:#1A1D23;}

/* Sidebar */
[data-testid="stSidebar"]{
  background:#FFFFFF !important;
  border-right:1px solid #E5E9F0 !important;
  box-shadow:2px 0 8px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] .block-container{padding:1rem !important;}
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
.top-title{font-size:1rem;font-weight:600;color:#1A1D23;margin:0;}
.top-sub{font-size:0.65rem;color:#2563EB;font-family:'JetBrains Mono',monospace;
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
.bub-u{background:#2563EB;color:#fff;border-radius:18px 18px 4px 18px;
       padding:0.65rem 1rem;max-width:70%;font-size:0.88rem;line-height:1.6;
       word-wrap:break-word;box-shadow:0 2px 6px rgba(37,99,235,0.2);}
.bub-b{background:#FFFFFF;color:#374151;border-radius:4px 18px 18px 18px;
       padding:0.65rem 1rem;max-width:82%;font-size:0.88rem;line-height:1.65;
       word-wrap:break-word;border:1px solid #E8EDF4;
       box-shadow:0 1px 3px rgba(0,0,0,0.05);}
.msg-meta{font-size:0.58rem;color:#9CA3AF;font-family:'JetBrains Mono',monospace;
          margin-top:0.18rem;padding:0 0.2rem;}
.msg-meta-r{text-align:right;}

/* Badges */
.badge{display:inline-block;font-size:0.56rem;font-weight:500;
       font-family:'JetBrains Mono',monospace;padding:0.08rem 0.38rem;
       border-radius:20px;border:1px solid;text-transform:uppercase;
       letter-spacing:0.04em;margin-left:0.25rem;vertical-align:middle;}
.b-text{color:#2563EB;border-color:rgba(37,99,235,0.25);background:rgba(37,99,235,0.07);}
.b-voice{color:#059669;border-color:rgba(5,150,105,0.25);background:rgba(5,150,105,0.07);}
.b-image{color:#7C3AED;border-color:rgba(124,58,237,0.25);background:rgba(124,58,237,0.07);}
.b-fusion{color:#D97706;border-color:rgba(217,119,6,0.25);background:rgba(217,119,6,0.07);}
.b-hi{color:#059669;border-color:rgba(5,150,105,0.25);background:rgba(5,150,105,0.07);}
.b-lo{color:#DC2626;border-color:rgba(220,38,38,0.25);background:rgba(220,38,38,0.07);}

/* KB card */
.kb-card{margin-top:0.6rem;background:#F0F4FF;border-radius:10px;
         padding:0.65rem 0.85rem;border-left:3px solid #2563EB;}
.kb-name{font-weight:600;color:#1E3A8A;margin-bottom:0.3rem;font-size:0.82rem;}
.kb-row{display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.2rem;}
.kb-item{font-size:0.67rem;color:#6B7280;font-family:'JetBrains Mono',monospace;}
.kb-item span{color:#374151;font-family:'Inter',sans-serif;}
.kb-events{margin-top:0.35rem;color:#B45309;font-size:0.73rem;}
.kb-ev-lbl{font-family:'JetBrains Mono',monospace;font-size:0.58rem;
           text-transform:uppercase;letter-spacing:0.08em;color:#9CA3AF;margin-bottom:0.1rem;}

/* Empty state */
.empty-state{text-align:center;padding:2rem 1rem;}
.empty-icon{font-size:2.2rem;margin-bottom:0.5rem;}
.empty-title{font-size:1rem;font-weight:600;color:#1A1D23;margin-bottom:0.3rem;}
.empty-sub{font-size:0.82rem;color:#6B7280;line-height:1.7;}

/* Input section */
.input-section{
  background:#FFFFFF;
  border:1.5px solid #E5E9F0;
  border-radius:14px;
  padding:0.5rem 0.75rem 0.5rem 0.75rem;
  margin-top:0.5rem;
  box-shadow:0 2px 8px rgba(0,0,0,0.05);
}
.input-section:focus-within{
  border-color:#2563EB;
  box-shadow:0 2px 8px rgba(37,99,235,0.1);
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
.attach-box{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
            padding:0.35rem 0.7rem;font-size:0.75rem;color:#1D4ED8;
            margin-bottom:0.4rem;display:flex;align-items:center;gap:0.4rem;}

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
.sb-sub{font-size:0.6rem;color:#2563EB;font-family:'JetBrains Mono',monospace;
        text-transform:uppercase;letter-spacing:0.08em;}
.sb-sec{font-size:0.6rem;font-family:'JetBrains Mono',monospace;color:#2563EB;
        text-transform:uppercase;letter-spacing:0.1em;margin:0.65rem 0 0.35rem 0;}
.sb-div{border-top:1px solid #F0F2F5;margin:0.65rem 0;}

/* Buttons */
.stButton>button{
  width:100%;background:linear-gradient(135deg,#2563EB,#1D4ED8) !important;
  color:#fff !important;border:none !important;border-radius:9px !important;
  padding:0.5rem !important;font-weight:600 !important;font-size:0.82rem !important;
  box-shadow:0 2px 5px rgba(37,99,235,0.25) !important;transition:all 0.15s !important;
}
.stButton>button:hover{opacity:0.9 !important;}
.sug .stButton>button{
  background:#F8FAFC !important;color:#4B5563 !important;
  border:1px solid #E5E9F0 !important;font-weight:400 !important;
  font-size:0.76rem !important;box-shadow:none !important;
}
.sug .stButton>button:hover{border-color:#2563EB !important;color:#2563EB !important;}
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
  background:#F8FAFC !important;border:1.5px solid #E5E9F0 !important;
  border-radius:8px !important;color:#1A1D23 !important;font-size:0.85rem !important;
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
.icon-btn .stButton>button{
  background:#F8FAFC !important;color:#6B7280 !important;
  border:1px solid #E5E9F0 !important;border-radius:8px !important;
  padding:0.3rem 0.5rem !important;font-size:1rem !important;
  font-weight:400 !important;box-shadow:none !important;
  min-height:32px !important;
}
.icon-btn .stButton>button:hover{
  border-color:#2563EB !important;color:#2563EB !important;
  background:#EFF6FF !important;
}
.icon-btn-active .stButton>button{
  background:#EFF6FF !important;color:#2563EB !important;
  border:1.5px solid #2563EB !important;border-radius:8px !important;
  padding:0.3rem 0.5rem !important;font-size:1rem !important;
  font-weight:400 !important;box-shadow:none !important;
}
.icon-btn-danger .stButton>button{
  background:#FEF2F2 !important;color:#EF4444 !important;
  border:1px solid #FECACA !important;border-radius:8px !important;
  padding:0.3rem 0.5rem !important;font-size:0.8rem !important;
  font-weight:400 !important;box-shadow:none !important;
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
       "voice+image":'<span class="badge b-fusion">🎤📷 fusion</span>'}
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
            st.markdown(
                f'<div class="msg-wrap">'
                f'<div class="msg-bot">'
                f'<div class="bot-av">🤖</div>'
                f'<div class="bub-b">{sc}{kb_card(r)}</div>'
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
    ⚡ <b>Combine</b> — image + text hint
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

# ── UNIFIED INPUT SECTION ─────────────────────────────────────────────
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

# Attachment preview
if st.session_state.pending_img:
    st.markdown(
        f"<div class='attach-box'>📷 <b>{_html.escape(st.session_state.pending_name)}</b> attached "
        f"<span style='color:#6B7280;font-size:0.68rem'>— will send with your message</span></div>",
        unsafe_allow_html=True)

# ── Text input row ────────────────────────────────────────────────────
q = st.chat_input("Ask anything about the campus…", key="chat_in")
if q:
    if st.session_state.pending_img:
        with st.spinner(""):
            do_image(st.session_state.pending_img, hint=q,
                     img_name=st.session_state.pending_name)
        st.session_state.pending_img  = None
        st.session_state.pending_name = ""
    else:
        do_text(q.strip(), "text")
    st.rerun()

# ── Modality toggle row ───────────────────────────────────────────────
ic1, ic2, ic3, ic4 = st.columns([1,1,1,2])
with ic1:
    st.markdown("<div class='sec'>", unsafe_allow_html=True)
    if st.button("🎤 Voice", key="tog_voice", use_container_width=True):
        st.session_state.show_voice = not st.session_state.show_voice
        st.session_state.show_image = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with ic2:
    st.markdown("<div class='sec'>", unsafe_allow_html=True)
    if st.button("📷 Image", key="tog_image", use_container_width=True):
        st.session_state.show_image = not st.session_state.show_image
        st.session_state.show_voice = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with ic3:
    if st.session_state.pending_img:
        st.markdown("<div class='sec'>", unsafe_allow_html=True)
        if st.button("❌ Remove", key="rm_img", use_container_width=True):
            st.session_state.pending_img  = None
            st.session_state.pending_name = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
with ic4:
    # Status indicator
    active_modes = []
    if st.session_state.show_voice: active_modes.append("🎤 Voice panel open")
    if st.session_state.show_image: active_modes.append("📷 Image panel open")
    if st.session_state.pending_img: active_modes.append("📎 Image attached")
    if active_modes:
        st.markdown(f"<div style='font-size:0.65rem;color:#2563EB;padding:0.3rem 0;"
                    f"font-family:JetBrains Mono,monospace;'>"
                    f"{' · '.join(active_modes)}</div>",
                    unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close input-section

# ── Voice panel (inline, below input) ────────────────────────────────
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
                        _,run_voice=load_pipelines()
                        with st.spinner("Transcribing…"):
                            suffix = st.session_state.get("audio_suffix", ".wav")
                            r=run_voice(st.session_state.audio_data, suffix=suffix)
                            tr=r.get("transcript","")
                            # If image also attached → voice+image fusion
                            if st.session_state.pending_img:
                                do_image(st.session_state.pending_img, hint=tr,
                                         img_name=st.session_state.pending_name)
                                st.session_state.pending_img=None
                                st.session_state.pending_name=""
                            else:
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

# ── Image panel (inline, below input) ────────────────────────────────
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
            if st.button("�� Identify & Ask", key="send_img2",
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
