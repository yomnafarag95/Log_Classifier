import streamlit as st
import joblib, numpy as np, pandas as pd, re, os, datetime, matplotlib, base64, io
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from collections import Counter
matplotlib.use("Agg")

st.set_page_config(page_title="Firewall Log Classifier", layout="wide", initial_sidebar_state="collapsed")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    m=joblib.load(os.path.join(MODEL_DIR,"model.joblib"))
    s=joblib.load(os.path.join(MODEL_DIR,"scaler.joblib"))
    l=joblib.load(os.path.join(MODEL_DIR,"label_encoder.joblib"))
    return m,s,l
model,scaler,le=load_artifacts()
AC={"allow":"#7ec8e3","deny":"#f4a261","drop":"#f9e07f","reset-both":"#b39ddb"}
for k,v in [("page","home"),("chat_history",[]),("last_result",None),("uploaded_results",[]),("file_name","")]:
    if k not in st.session_state: st.session_state[k]=v

def parse_log_line(line):
    nums=[int(n) for n in re.findall(r"\b(\d+)\b",line)]; nums+=[0]*(11-len(nums)); return nums[:11]
def predict(features):
    a=np.array(features,dtype=float).reshape(1,-1); sc=scaler.transform(a)
    lbl=model.predict(sc)[0]; pr=model.predict_proba(sc)[0]
    act=le.inverse_transform([lbl])[0]; conf=float(np.max(pr))
    pd_={le.inverse_transform([i])[0]:round(float(p)*100,1) for i,p in enumerate(pr)}
    return act,conf,pd_
def bc(a): return AC.get(a,"#888")
def add_hist(log,action,conf):
    st.session_state.chat_history.append({"time":datetime.datetime.now().strftime("%H:%M:%S"),"log":log[:80]+("..." if len(log)>80 else ""),"action":action,"confidence":conf})
def fig_to_b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",transparent=True); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def lfig(w=2.5,h=1.5):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_alpha(0); ax.set_facecolor((0.94,0.97,1.0,0.5))
    ax.tick_params(colors="#4a7a9a",labelsize=5); ax.xaxis.label.set_color("#4a7a9a"); ax.yaxis.label.set_color("#4a7a9a")
    for sp in ax.spines.values(): sp.set_edgecolor("#b0d8f0"); sp.set_linewidth(0.6)
    return fig,ax

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Outfit:wght@300;400;500;600&display=swap');
*,html,body,[class*="css"]{font-family:'Outfit',sans-serif;box-sizing:border-box;}
.stApp{background:radial-gradient(ellipse at 15% 60%,#8ecfee 0%,#b0dcf5 25%,#d8eefa 45%,#eef6fc 65%,#ffffff 80%,#cce8f5 100%);background-attachment:fixed;min-height:100vh;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0.5rem 2rem 0.2rem 2rem !important;max-width:100% !important;}
section[data-testid="stSidebar"]{display:none !important;}
[data-testid="collapsedControl"]{display:none !important;}

/* ── Greeting ── */
.greeting-glow{position:relative;text-align:center;margin-bottom:36px;animation:floatUp 0.85s cubic-bezier(0.16,1,0.3,1) both;}
.greeting-glow::before{content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-55%);width:600px;height:320px;background:radial-gradient(ellipse at center,rgba(255,255,255,0.82) 0%,rgba(255,255,255,0.45) 45%,transparent 75%);border-radius:50%;pointer-events:none;z-index:0;}
.greeting-glow>*{position:relative;z-index:1;}
.greeting-hi{font-family:'Cormorant Garamond',serif;font-size:clamp(2.4rem,5vw,3.8rem);font-weight:300;color:#1a4a72;line-height:1.25;margin-bottom:8px;}
.greeting-hi strong{font-weight:600;color:#0a7ab5;}
.greeting-sub{font-size:0.95rem;color:#6a9abb;font-weight:300;}
@keyframes floatUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}

/* ── Textarea ── */
.stTextArea textarea{border:1px solid rgba(140,200,235,0.35) !important;border-radius:16px !important;background:white !important;font-size:0.97rem !important;color:#2a4060 !important;padding:20px 22px !important;resize:none !important;box-shadow:0 8px 32px rgba(80,160,220,0.10) !important;outline:none !important;}
.stTextArea textarea:focus{border-color:rgba(74,176,232,0.45) !important;outline:none !important;}
.stTextArea textarea::placeholder{color:#aac8de !important;font-family:'Cormorant Garamond',serif !important;font-style:italic !important;}
.stTextArea label{display:none !important;}
.stTextArea,.stTextArea>div,.stTextArea>div>div{border:none !important;box-shadow:none !important;background:transparent !important;outline:none !important;}
[data-testid="stFileUploader"]{visibility:hidden !important;height:0 !important;overflow:hidden !important;margin:0 !important;padding:0 !important;}

/* ── All Streamlit buttons: transparent, white border, Cormorant, #1a4a72 ── */
div.stButton > button,
div.stButton > button:hover,
div.stButton > button:focus,
div.stButton > button:active {
    background: transparent !important;
    background-color: transparent !important;
    border: 1.5px solid rgba(255,255,255,0.85) !important;
    border-radius: 12px !important;
    color: #1a4a72 !important;
    -webkit-text-fill-color: #1a4a72 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 11px 0 !important;
    width: 100% !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
}
div.stButton > button:hover {
    background: rgba(255,255,255,0.22) !important;
    border-color: white !important;
}
div.stButton > button p {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    color: #1a4a72 !important;
    -webkit-text-fill-color: #1a4a72 !important;
    letter-spacing: 0.06em !important;
    margin: 0 !important;
}

/* ── Button columns: force full width at every level ── */
[data-testid="stHorizontalBlock"] { gap: 6px !important; }
div.stButton { width: 100% !important; display: block !important; }
div.stButton > button { display: block !important; }
.metric-card{background:rgba(255,255,255,0.50);backdrop-filter:blur(6px);border:1px solid rgba(180,220,245,0.55);border-radius:14px;padding:12px 10px;text-align:center;}
.metric-card .label{font-size:0.63rem;color:#90b8d4;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:7px;font-weight:600;}
.metric-card .value{font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:600;color:#1a7ab5;line-height:1;}
.metric-card .sub{font-size:0.7rem;color:#90b8d4;margin-top:5px;}
.result-card{background:rgba(255,255,255,0.55);backdrop-filter:blur(8px);border-radius:16px;padding:18px 24px;box-shadow:0 4px 20px rgba(80,160,220,0.10);border:1px solid rgba(180,220,245,0.50);margin-top:16px;animation:floatUp 0.7s cubic-bezier(0.16,1,0.3,1) both;}
.sec-head{font-size:0.64rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#7ab0d0;border-bottom:1px solid rgba(140,200,235,0.26);padding-bottom:7px;margin-bottom:14px;margin-top:28px;}
.chat-item{background:white;border:1px solid rgba(140,200,235,0.26);border-left:3px solid #4ab0e8;border-radius:0 12px 12px 0;padding:14px 20px;margin:8px 0;font-size:0.92rem;color:#2a4060;}
.chat-item .timestamp{font-size:0.68rem;letter-spacing:0.10em;text-transform:uppercase;color:#90b8d4;margin-bottom:4px;font-weight:600;}
.chat-action-badge{display:inline-block;padding:2px 12px;border-radius:20px;font-size:0.74rem;letter-spacing:0.10em;text-transform:uppercase;font-weight:600;margin-top:6px;}
.chart-card{background:rgba(255,255,255,0.55);backdrop-filter:blur(10px);border:1px solid rgba(160,215,245,0.55);border-radius:16px;padding:10px 12px 6px;box-shadow:0 4px 18px rgba(80,160,220,0.10),inset 0 1px 0 rgba(255,255,255,0.8);}
.chart-title{font-size:0.6rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#5a9abf;margin-bottom:4px;}
.stDataFrame{border:1px solid rgba(140,200,235,0.30) !important;border-radius:12px !important;overflow:hidden !important;}
.stDataFrame > div,.stDataFrame [data-testid="stDataFrameResizable"]{background:white !important;border-radius:12px !important;}
.stDataFrame iframe{background:white !important;color-scheme:light !important;}
[data-testid="stImage"],[data-testid="stpyplot"]{margin:0 !important;padding:0 !important;}
.element-container{margin-bottom:0 !important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-thumb{background:rgba(74,176,232,0.35);border-radius:10px;}
</style>
""", unsafe_allow_html=True)


# ── HOME ──────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("<div style='height:14vh'></div>", unsafe_allow_html=True)
    _, cx, _ = st.columns([1, 2.4, 1])
    with cx:
        st.markdown("""<div class="greeting-glow">
            <div class="greeting-hi">Hi, I am <strong>Log Classifier</strong><br>How can I help you today?</div>
        </div>""", unsafe_allow_html=True)

        log_input = st.text_area("log", placeholder="Ask me anything...", height=160,
                                  key="log_text_input", label_visibility="collapsed")

        uploaded_file = st.file_uploader("u", type=["txt","csv","log"],
                                          label_visibility="collapsed", key="home_uploader")
        if uploaded_file:
            content=uploaded_file.read().decode("utf-8",errors="ignore")
            lines=[l.strip() for l in content.splitlines() if l.strip()]; results=[]
            for line in lines:
                try:
                    feats=parse_log_line(line); act,conf,probs=predict(feats)
                    results.append({"log":line[:60],"action":act,"confidence":round(conf*100,1),"probabilities":probs})
                except: pass
            st.session_state.update({"uploaded_results":results,"file_name":uploaded_file.name,"page":"dashboard"}); st.rerun()

        # All 5 buttons in one row, spanning the full width of the input box
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: new_chat_btn   = st.button("New Chat",      key="btn_new_chat")
        with c2: history_btn    = st.button("Chat History",  key="btn_chat_history")
        with c3: analyze_btn    = st.button("Analyze",       key="btn_analyze")
        with c4: upload_btn     = st.button("Upload",        key="btn_upload")
        with c5: dashboard_btn  = st.button("Dashboard",     key="btn_dashboard")

        if new_chat_btn:
            st.session_state.update({"chat_history":[],"last_result":None,"uploaded_results":[],"file_name":"","page":"home"}); st.rerun()
        if history_btn:
            st.session_state.page = "history"; st.rerun()

        if analyze_btn:
            if log_input.strip():
                try:
                    feats=parse_log_line(log_input); act,conf,probs=predict(feats)
                    st.session_state.last_result={"log":log_input,"action":act,"confidence":round(conf*100,1),"probabilities":probs,"features":feats}
                    add_hist(log_input,act,round(conf*100,1)); st.rerun()
                except Exception as e: st.error(f"Prediction error: {e}")
            else: st.warning("Please paste a log entry before analyzing.")
        if upload_btn: st.info("Drag and drop a file above.")
        if dashboard_btn: st.session_state.page="dashboard"; st.rerun()

        if st.session_state.last_result:
            r=st.session_state.last_result; color=bc(r["action"])
            pb="".join([f'<span style="display:inline-block;margin-right:12px;font-size:0.8rem;color:#4a7a9a;"><span style="color:{bc(k)};font-weight:600;">{k}</span> {v}%</span>' for k,v in r["probabilities"].items()])
            st.markdown(f'''<div class="result-card">
                <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#7ab0d0;margin-bottom:10px;">Prediction Result</div>
                <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">
                    <span style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;font-weight:600;color:{color};">{r["action"].upper()}</span>
                    <span style="font-size:0.85rem;color:#6a9abb;font-weight:300;">{r["confidence"]}% confidence</span>
                </div>
                <div style="padding-top:6px;border-top:1px solid rgba(140,200,235,0.25);">{pb}</div>
            </div>''', unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────
elif st.session_state.page == "history":
    st.markdown("<div style='height:14vh'></div>", unsafe_allow_html=True)
    _, hcol, _ = st.columns([1, 2.4, 1])
    with hcol:
        st.markdown('<div class="back-btn" style="margin-bottom:18px">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="hist_back"): st.session_state.page="home"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head" style="margin-top:0">Chat History</div>', unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown('<div class="chat-item">No history yet in this session.</div>', unsafe_allow_html=True)
        else:
            for item in reversed(st.session_state.chat_history):
                color=bc(item["action"])
                st.markdown(f'<div class="chat-item"><div class="timestamp">{item["time"]}</div><div style="margin-bottom:6px;">{item["log"]}</div><span class="chat-action-badge" style="background:{color}20;color:{color};border:1px solid {color}50;">{item["action"].upper()}</span><span style="font-size:0.76rem;color:#90b8d4;margin-left:10px;">{item["confidence"]}% confidence</span></div>', unsafe_allow_html=True)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
elif st.session_state.page == "dashboard":
    hd1,hd2=st.columns([6,1])
    with hd1: st.markdown('<div style=\'font-family:"Cormorant Garamond",serif;font-size:1.5rem;font-weight:400;color:#1a4a72;margin-bottom:4px;\'>Dashboard <span style="font-size:0.58rem;color:#7ab0d0;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;vertical-align:middle;margin-left:6px;">Firewall Log Classification</span></div>', unsafe_allow_html=True)
    with hd2:
        if st.button("← Home", key="dash_back"): st.session_state.page="home"; st.rerun()

    all_results=[]
    if st.session_state.last_result: all_results.append(st.session_state.last_result)
    all_results+=list(st.session_state.uploaded_results)

    if not all_results:
        st.info("No predictions yet.")
    else:
        actions=[r["action"] for r in all_results]; count=Counter(actions); labels=["allow","deny","drop","reset-both"]
        total=len(all_results)
        allow_n=count.get("allow",0); block_n=count.get("deny",0)+count.get("drop",0); reset_n=count.get("reset-both",0)
        allow_pct=round(allow_n/total*100) if total else 0
        block_pct=round(block_n/total*100) if total else 0

        # ── Row 1: metric cards ──
        card_style="background:rgba(255,255,255,0.60);backdrop-filter:blur(8px);border-radius:12px;padding:7px 12px;box-shadow:0 3px 14px rgba(80,160,220,0.08),inset 0 1px 0 rgba(255,255,255,0.9);"
        lbl_style="font-size:0.50rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#7ab0d0;margin-bottom:2px;"
        sub_style="font-size:0.55rem;color:#90b8d4;margin-top:2px;"
        m1,m2,m3,m4=st.columns(4)
        m1.markdown(f'<div style="{card_style}border:1px solid rgba(160,215,245,0.55);"><div style="{lbl_style}">Total</div><div style="font-family:\'Cormorant Garamond\',serif;font-size:1.6rem;font-weight:600;color:#1a4a72;line-height:1;">{total}</div><div style="{sub_style}">entries classified</div></div>',unsafe_allow_html=True)
        m2.markdown(f'<div style="{card_style}border:1px solid rgba(126,200,227,0.45);"><div style="{lbl_style}">Allow</div><div style="font-family:\'Cormorant Garamond\',serif;font-size:1.6rem;font-weight:600;color:#7ec8e3;line-height:1;">{allow_n}</div><div style="{sub_style}">{allow_pct}% of traffic</div></div>',unsafe_allow_html=True)
        m3.markdown(f'<div style="{card_style}border:1px solid rgba(244,162,97,0.35);"><div style="{lbl_style}">Blocked</div><div style="font-family:\'Cormorant Garamond\',serif;font-size:1.6rem;font-weight:600;color:#f4a261;line-height:1;">{block_n}</div><div style="{sub_style}">{block_pct}% of traffic</div></div>',unsafe_allow_html=True)
        m4.markdown(f'<div style="{card_style}border:1px solid rgba(179,157,219,0.40);"><div style="{lbl_style}">Reset-Both</div><div style="font-family:\'Cormorant Garamond\',serif;font-size:1.6rem;font-weight:600;color:#b39ddb;line-height:1;">{reset_n}</div><div style="{sub_style}">terminated</div></div>',unsafe_allow_html=True)

        # ── Row 2: charts ──
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div style="background:rgba(255,255,255,0.60);backdrop-filter:blur(10px);border:1px solid rgba(160,215,245,0.50);border-radius:14px;padding:7px 12px 4px;box-shadow:0 3px 14px rgba(80,160,220,0.08);"><div style="font-size:0.50rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#5a9abf;margin-bottom:3px;">Action Distribution</div>', unsafe_allow_html=True)
            fig_p,ax_p=plt.subplots(figsize=(2.8,1.7))
            fig_p.patch.set_alpha(0); ax_p.set_facecolor((0,0,0,0))
            non_zero=[(count.get(l,0),bc(l),l) for l in labels if count.get(l,0)>0]
            if non_zero:
                s_,c_,l_=zip(*non_zero)
                wedges,_,at=ax_p.pie(s_,labels=None,colors=c_,autopct="%1.0f%%",startangle=90,
                    pctdistance=0.68,wedgeprops={"linewidth":2,"edgecolor":"white","width":0.48})
                for t in at: t.set_fontsize(6.5); t.set_color("white"); t.set_fontweight("bold")
                for w in wedges: w.set_alpha(0.90)
                ax_p.text(0,0,f"{total}\nlogs",ha="center",va="center",fontsize=7.5,color="#1a4a72",fontweight="bold",linespacing=1.3)
                all_patches=[mpatches.Patch(color=bc(l),alpha=0.9,label=f"{l.upper()} {count.get(l,0)}") for l in labels]
                ax_p.legend(handles=all_patches,loc="lower center",bbox_to_anchor=(0.5,-0.06),ncol=2,fontsize=5.5,framealpha=0,labelcolor="#2a5a82")
            fig_p.tight_layout(pad=0.1)
            st.pyplot(fig_p, use_container_width=False); plt.close(fig_p)
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div style="background:rgba(255,255,255,0.60);backdrop-filter:blur(10px);border:1px solid rgba(160,215,245,0.50);border-radius:14px;padding:7px 12px 4px;box-shadow:0 3px 14px rgba(80,160,220,0.08);"><div style="font-size:0.50rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#5a9abf;margin-bottom:3px;">Confidence per Entry <span style="color:#4ab0e8;font-weight:400;">— 80% threshold</span></div>', unsafe_allow_html=True)
            fig_b,ax_b=plt.subplots(figsize=(3.4,1.7))
            fig_b.patch.set_alpha(0); ax_b.set_facecolor((0.95,0.97,1.0,0.5))
            confs=[r["confidence"] for r in all_results[-12:]]; bar_c=[bc(r["action"]) for r in all_results[-12:]]
            x_pos=list(range(len(confs)))
            bars=ax_b.bar(x_pos,confs,color=bar_c,edgecolor="white",linewidth=0.8,width=0.62,alpha=0.88)
            ax_b.axhline(y=80,color="#4ab0e8",linewidth=0.8,linestyle="--",alpha=0.8)
            ax_b.set_ylim(0,120); ax_b.set_yticks([0,50,100]); ax_b.set_yticklabels(["0%","50%","100%"],fontsize=5.5)
            ax_b.set_xticks(x_pos); ax_b.set_xticklabels([str(i+1) for i in x_pos],fontsize=5.5)
            ax_b.tick_params(colors="#4a7a9a",length=2); ax_b.grid(axis="y",color="#d0eaf8",linewidth=0.5,alpha=0.9)
            for sp in ax_b.spines.values(): sp.set_edgecolor("#c8e0f0"); sp.set_linewidth(0.5)
            for b,v in zip(bars,confs): ax_b.text(b.get_x()+b.get_width()/2,v+1,f"{v:.0f}%",ha="center",color="#2a5a82",fontsize=4.5,fontweight="bold")
            seen={}
            for r in all_results:
                if r["action"] not in seen: seen[r["action"]]=bc(r["action"])
            ax_b.legend([mpatches.Patch(color=c,alpha=0.88) for c in seen.values()],[k.upper() for k in seen],
                loc="upper right",fontsize=5,framealpha=0.5,labelcolor="#2a5a82",edgecolor="#d0e8f4")
            fig_b.tight_layout(pad=0.2)
            st.pyplot(fig_b, use_container_width=False); plt.close(fig_b)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 3: white HTML table ──
        rows_html="".join([f'<tr><td style="padding:5px 10px;font-size:0.78rem;color:#2a4060;border-bottom:1px solid rgba(200,230,248,0.5);max-width:500px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{"("+r["log"][:65]+")" if not isinstance(r["log"],str) else r["log"][:65]}</td><td style="padding:5px 10px;font-size:0.75rem;font-weight:600;color:{bc(r["action"])};border-bottom:1px solid rgba(200,230,248,0.5);white-space:nowrap;">{r["action"].upper()}</td><td style="padding:5px 10px;font-size:0.75rem;color:#4a7a9a;border-bottom:1px solid rgba(200,230,248,0.5);white-space:nowrap;">{r["confidence"]}%</td></tr>' for r in all_results])
        st.markdown(f"""
        <div style="background:white;border:1px solid rgba(140,200,235,0.35);border-radius:12px;overflow:hidden;margin-top:6px;">
            <table style="width:100%;border-collapse:collapse;background:white;">
                <thead><tr style="background:rgba(220,240,252,0.85);">
                    <th style="padding:6px 10px;font-size:0.60rem;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;color:#1a4a72;text-align:left;">Log</th>
                    <th style="padding:6px 10px;font-size:0.60rem;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;color:#1a4a72;text-align:left;">Action</th>
                    <th style="padding:6px 10px;font-size:0.60rem;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;color:#1a4a72;text-align:left;">Confidence</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>""", unsafe_allow_html=True)

# ── FORCE STYLES via JavaScript — runs after React renders, always wins ───────
st.markdown("""
<script>
(function() {
    function styleBtn(btn) {
        var s = btn.style;
        s.setProperty('background',         'transparent',                   'important');
        s.setProperty('background-color',   'transparent',                   'important');
        s.setProperty('border',             '1.5px solid rgba(255,255,255,0.85)', 'important');
        s.setProperty('border-radius',      '10px',                          'important');
        s.setProperty('display',            'block',                         'important');
        s.setProperty('color',              '#1a4a72',                       'important');
        s.setProperty('-webkit-text-fill-color', '#1a4a72',                  'important');
        s.setProperty('font-family',        "'Cormorant Garamond', serif",   'important');
        s.setProperty('font-size',          '1.2rem',                        'important');
        s.setProperty('font-weight',        '500',                           'important');
        s.setProperty('letter-spacing',     '0.06em',                        'important');
        s.setProperty('box-shadow',         'none',                          'important');
        s.setProperty('outline',            'none',                          'important');
        s.setProperty('width',              '100%',                          'important');
        s.setProperty('text-align',         'center',                        'important');
        s.setProperty('padding',            '11px 0',                        'important');
        s.setProperty('cursor',             'pointer',                       'important');
        // Force parent div.stButton to full width too
        if (btn.parentElement) {
            btn.parentElement.style.setProperty('width',   '100%', 'important');
            btn.parentElement.style.setProperty('display', 'block','important');
        }
        // Style the <p> inside (Streamlit wraps text in <p>)
        var p = btn.querySelector('p');
        if (p) {
            p.style.setProperty('font-family',   "'Cormorant Garamond', serif", 'important');
            p.style.setProperty('font-size',      '1.2rem',    'important');
            p.style.setProperty('font-weight',    '500',        'important');
            p.style.setProperty('color',          '#1a4a72',    'important');
            p.style.setProperty('-webkit-text-fill-color', '#1a4a72', 'important');
            p.style.setProperty('letter-spacing', '0.06em',    'important');
            p.style.setProperty('margin',         '0',          'important');
        }
        btn.addEventListener('mouseenter', function(){
            btn.style.setProperty('background','rgba(255,255,255,0.22)','important');
            btn.style.setProperty('border-color','white','important');
        });
        btn.addEventListener('mouseleave', function(){
            btn.style.setProperty('background','transparent','important');
            btn.style.setProperty('border','1.5px solid rgba(255,255,255,0.85)','important');
        });
    }

    function applyAll() {
        document.querySelectorAll('div.stButton > button').forEach(styleBtn);
    }

    applyAll();
    [200, 500, 1000, 2000].forEach(function(d){ setTimeout(applyAll, d); });

    var observer = new MutationObserver(function(mutations) {
        var needsUpdate = false;
        mutations.forEach(function(m) {
            if (m.addedNodes.length) needsUpdate = true;
        });
        if (needsUpdate) applyAll();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // Force dataframe iframe white
    function styleDataframes() {
        document.querySelectorAll('.stDataFrame iframe').forEach(function(iframe) {
            try {
                var doc = iframe.contentDocument || iframe.contentWindow.document;
                if (!doc) return;
                var s = doc.createElement('style');
                s.textContent = 'body,html,table,thead,tbody,tr,th,td,div{background:white !important;color:#2a4060 !important;} thead th,thead td{background:rgba(220,240,252,0.95) !important;color:#1a4a72 !important;font-weight:700 !important;text-transform:uppercase !important;font-size:0.7rem !important;letter-spacing:0.06em !important;} tr:nth-child(even) td{background:rgba(235,247,255,0.7) !important;}';
                doc.head.appendChild(s);
            } catch(e) {}
        });
    }
    [500,1000,2000,3000].forEach(function(d){ setTimeout(styleDataframes, d); });
})();
</script>
""", unsafe_allow_html=True)
