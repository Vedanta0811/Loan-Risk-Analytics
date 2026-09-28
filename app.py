import os
from io import BytesIO
import joblib
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title='LoanLens AI', page_icon='🏦', layout='wide')
st.markdown('''<style>
.stApp{background:#eef8ff}.block-container{max-width:1200px;padding-top:5.2rem!important;padding-bottom:3rem}.stButton{overflow:visible!important}header[data-testid='stHeader']{background:rgba(238,248,255,.96)!important}.stButton>button{overflow:visible!important}.brand{font-size:25px;font-weight:800;color:#0f4c81}.brand span,.blue{color:#2386c8}.hero{font-size:56px;line-height:1.05;font-weight:850;color:#12324a}.hero span{color:#2386c8}.sub{color:#61798a;font-size:17px;line-height:1.6}.section{font-size:31px;font-weight:800;color:#12324a;margin-top:28px}.card,.metric,.factor{background:linear-gradient(145deg,#ffffff,#f4faff);border:1px solid #d8eaf5;border-radius:18px;padding:22px;box-shadow:5px 6px 14px rgba(30,90,120,.07),-4px -4px 10px rgba(255,255,255,.9);transition:all .25s ease}.card:hover,.metric:hover,.factor:hover{transform:translateY(-4px);box-shadow:7px 11px 22px rgba(30,90,120,.12),-4px -4px 12px rgba(255,255,255,.95)}.metric{text-align:center}.num{font-size:29px;font-weight:800;color:#12679b}.label,.muted{color:#718693;font-size:13px}.approved{background:linear-gradient(145deg,#effdf6,#e5f8ee);border:1px solid #a9e4c8;border-radius:20px;padding:28px;box-shadow:5px 8px 20px rgba(35,130,90,.08)}.rejected{background:linear-gradient(145deg,#fff7f7,#fff0f0);border:1px solid #f1b7b7;border-radius:20px;padding:28px;box-shadow:5px 8px 20px rgba(160,60,60,.08)}.result{font-size:34px;font-weight:850;color:#17394d}.prob{font-size:46px;font-weight:850;color:#12679b}.footer{text-align:center;color:#7a8d98;font-size:13px;padding:30px}div.stButton>button,div[data-testid='stDownloadButton'] button{border-radius:11px;font-weight:700;min-height:44px;height:44px;line-height:1.2;padding:8px 16px;margin:0;box-shadow:0 4px 8px rgba(35,100,140,.12);transition:all .2s ease}div.stButton>button:hover,div[data-testid='stDownloadButton'] button:hover{transform:translateY(-2px);box-shadow:0 7px 14px rgba(35,100,140,.18)}

.perf-card{background:linear-gradient(145deg,#ffffff,#f5faff);border:1px solid #d6e8f3;border-radius:20px;padding:24px;box-shadow:6px 8px 20px rgba(30,90,120,.07),-4px -4px 12px rgba(255,255,255,.9);margin-bottom:18px}
.perf-card-head{display:flex;justify-content:space-between;align-items:center;gap:15px;margin-bottom:18px}
.perf-title{font-size:21px;font-weight:850;color:#17394d}.perf-subtitle{font-size:13px;color:#7a8d98;margin-top:4px}
.model-pill,.importance-badge{background:#e7f4fd;color:#12679b;border:1px solid #c7e5f5;border-radius:999px;padding:7px 13px;font-size:12px;font-weight:800;white-space:nowrap}
.comparison-table-wrap{overflow:hidden;border:1px solid #dbeaf3;border-radius:14px}.comparison-table{width:100%;border-collapse:collapse;font-size:14px}.comparison-table th{background:#12679b;color:white;text-align:left;padding:14px 16px;font-weight:800}.comparison-table td{padding:14px 16px;border-bottom:1px solid #e7f0f5;color:#536b79;background:white}.comparison-table tr:nth-child(even) td{background:#f8fcff}.comparison-table tr:last-child td{border-bottom:none}.comparison-table .metric-name{font-weight:800;color:#17394d}.rf-value{display:inline-block;background:#e9f7ff;color:#12679b;border:1px solid #c9e8f6;border-radius:8px;padding:5px 9px;font-weight:850}.table-note{font-size:12px;color:#8497a2;margin-top:10px}
.legend-row{display:flex;gap:22px;justify-content:flex-end;color:#687f8d;font-size:12px;margin-bottom:16px}.legend-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:#9fb6c4}.rf-dot{background:#2386c8}.perf-bar-row{padding:15px 0;border-top:1px solid #edf3f7}.perf-bar-row:first-of-type{border-top:none}.perf-bar-label{font-weight:800;color:#17394d;font-size:14px;margin-bottom:9px}.bar-line{display:grid;grid-template-columns:28px 1fr 58px;align-items:center;gap:8px;margin:6px 0}.bar-model{font-size:11px;font-weight:800;color:#7a8d98}.bar-track{height:9px;background:#eaf2f6;border-radius:99px;overflow:hidden}.bar-fill{height:100%;border-radius:99px}.lr-fill{background:#9fb6c4}.rf-fill{background:#2386c8}.bar-number{font-size:12px;font-weight:800;color:#506b7a;text-align:right}
.cm-card{background:linear-gradient(145deg,#ffffff,#f5faff);border:1px solid #d6e8f3;border-radius:20px;padding:22px;box-shadow:6px 8px 20px rgba(30,90,120,.07),-4px -4px 12px rgba(255,255,255,.9)}.cm-title{font-size:19px;font-weight:850;color:#17394d}.cm-subtitle{font-size:12px;color:#8497a2;margin:4px 0 17px}.cm-grid{display:grid;grid-template-columns:82px 1fr 1fr;gap:6px}.cm-label{display:flex;align-items:center;justify-content:center;text-align:center;font-size:11px;font-weight:800;color:#728793;padding:7px}.cm-label.side{justify-content:flex-start}.cm-cell{min-height:90px;border-radius:12px;padding:12px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border:1px solid #dceaf2}.cm-cell small{font-size:10px;color:#718693;margin-bottom:5px}.cm-cell strong{font-size:25px;color:#17394d}.cm-cell.tn{background:#edf9f3;border-color:#c5ead6}.cm-cell.tp{background:#e8f6fd;border-color:#c4e4f3}.cm-cell.fp{background:#fff5ea;border-color:#f2d7b4}.cm-cell.fn{background:#fff1f1;border-color:#efc5c5}
.fi-row{display:grid;grid-template-columns:155px 1fr 65px;align-items:center;gap:14px;padding:13px 0;border-bottom:1px solid #edf3f7}.fi-row:last-of-type{border-bottom:none}.fi-name{font-size:13px;font-weight:750;color:#526d7b}.fi-track{height:13px;background:#e9f2f6;border-radius:99px;overflow:hidden}.fi-fill{height:100%;background:linear-gradient(90deg,#2386c8,#5ba9d7);border-radius:99px}.fi-value{font-size:13px;font-weight:850;color:#12679b;text-align:right}
@media(max-width:700px){.perf-card-head{align-items:flex-start;flex-direction:column}.fi-row{grid-template-columns:120px 1fr 55px}.cm-grid{grid-template-columns:65px 1fr 1fr}.legend-row{justify-content:flex-start}.comparison-table{font-size:12px}.comparison-table th,.comparison-table td{padding:11px 9px}}
</style>''', unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page='home'
if 'last' not in st.session_state: st.session_state.last=None

@st.cache_resource
def load_model():
    p=os.path.join('models','loan_approval_random_forest.pkl')
    if not os.path.exists(p): st.error('Model not found: models/loan_approval_random_forest.pkl'); st.stop()
    return joblib.load(p)
model=load_model()

def explain(r):
    explanations = []
    credit = int(r["Credit_Score"])
    income = float(r["Annual_Income"])
    loan = float(r["Loan_Amount_Requested"])
    debt = float(r["Outstanding_Debt"])
    default_risk = float(r["Default_Risk"])
    if credit >= 750:
        explanations.append(f"Credit Score of {credit} is relatively high and is an important model input.")
    elif credit >= 650:
        explanations.append(f"Credit Score of {credit} is in the moderate-to-good range and contributes to the prediction.")
    elif credit >= 550:
        explanations.append(f"Credit Score of {credit} is relatively low and may reduce the predicted approval probability.")
    else:
        explanations.append(f"Credit Score of {credit} is low and may strongly affect the model prediction.")
    ratio = loan / income if income else 999
    if ratio <= 0.25:
        explanations.append("The requested loan amount is relatively low compared with annual income.")
    elif ratio <= 0.50:
        explanations.append("The requested loan amount is moderate relative to annual income.")
    else:
        explanations.append("The requested loan amount is relatively high compared with annual income.")
    if debt <= 10000:
        explanations.append("Outstanding debt is relatively low.")
    elif debt >= 20000:
        explanations.append("Outstanding debt is relatively high.")
    if default_risk <= 0.30:
        explanations.append("The recorded default-risk value is relatively low.")
    elif default_risk >= 0.70:
        explanations.append("The recorded default-risk value is relatively high.")
    explanations.append("The final prediction is generated from the combined pattern of all input features, not a single rule.")
    return explanations

def pdf(a):
    b=BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm)
    s=getSampleStyleSheet(); title=ParagraphStyle('t',parent=s['Title'],fontSize=24,alignment=TA_CENTER,textColor=colors.HexColor('#12324a')); h=ParagraphStyle('h',parent=s['Heading2'],fontSize=15,textColor=colors.HexColor('#12679b')); n=ParagraphStyle('n',parent=s['Normal'],fontSize=10,leading=15,textColor=colors.HexColor('#334e5e'))
    story=[Paragraph('LoanLens AI',title),Paragraph('Loan Approval Assessment Report',ParagraphStyle('sub',parent=n,alignment=TA_CENTER)),Spacer(1,15)]
    story += [Table([['Prediction',a['status']],['Approval Probability',f"{a['probability']:.2f}%"]],colWidths=[65*mm,90*mm],style=TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#eef8ff')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d8eaf5')),('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)])),Spacer(1,12),Paragraph('Applicant Details',h)]
    rows=[['Field','Value'],['Gender',a['Gender']],['Age',str(a['Age'])],['Marital Status',a['Marital_Status']],['Dependents',str(a['Dependents'])],['Education',a['Education']],['Employment Status',a['Employment_Status']],['Occupation Type',a['Occupation_Type']],['Residential Status',a['Residential_Status']],['City/Town',a['City/Town']],['Annual Income',f"₹{a['Annual_Income']:,.0f}"],['Monthly Expenses',f"₹{a['Monthly_Expenses']:,.0f}"],['Credit Score',str(a['Credit_Score'])],['Existing Loans',str(a['Existing_Loans'])],['Existing Loan Amount',f"₹{a['Total_Existing_Loan_Amount']:,.0f}"],['Outstanding Debt',f"₹{a['Outstanding_Debt']:,.0f}"],['Loan History',str(a['Loan_History'])],['Loan Amount Requested',f"₹{a['Loan_Amount_Requested']:,.0f}"],['Loan Term',f"{a['Loan_Term']} months"],['Loan Purpose',a['Loan_Purpose']],['Interest Rate',f"{a['Interest_Rate']:.2f}%"],['Loan Type',a['Loan_Type']],['Co-Applicant',a['Co-Applicant']],['Bank Account History',str(a['Bank_Account_History'])],['Transaction Frequency',str(a['Transaction_Frequency'])],['Default Risk',f"{a['Default_Risk']:.2f}"]]
    story += [Table(rows,colWidths=[65*mm,90*mm],repeatRows=1,style=TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#12679b')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d8eaf5')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fcff')]),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])),Spacer(1,12),Paragraph('Prediction Explanation',h)]
    story += [Paragraph('• '+z,n) for z in a['explanation']]+[Spacer(1,8),Paragraph('Model Information',h),Paragraph('Prediction generated using the deployed Random Forest classifier. This is a decision-support prototype and does not replace regulatory checks, bank policies, or human review.',n)]
    doc.build(story); return b.getvalue()

def nav():
    c1,c2=st.columns([.25,.75]);
    with c1:
        if st.button('← Home',use_container_width=True): st.session_state.page='home'; st.rerun()
    with c2: st.markdown('<div style="text-align:right" class="brand">LoanLens <span>AI</span></div>',unsafe_allow_html=True)

def home():
    st.markdown('<div class="brand">LoanLens <span>AI</span></div>',unsafe_allow_html=True); st.write('')
    a,b=st.columns([1.15,.85])
    with a:
        st.markdown('<div class="hero">Understand your<br><span>loan approval</span> outlook.</div>',unsafe_allow_html=True)
        st.markdown('<p class="sub">LoanLens AI uses a trained Random Forest model to estimate loan approval outcomes from applicant and financial information.</p>',unsafe_allow_html=True)
        x,y=st.columns(2)
        with x:
            if st.button('Start Assessment →',type='primary',use_container_width=True):st.session_state.page='assessment';st.rerun()
        with y:
            if st.button('View Model Performance',use_container_width=True):st.session_state.page='performance';st.rerun()
    with b:
        st.markdown('<div class="card"><div class="muted">SAMPLE ASSESSMENT</div><h1 style="color:#12324a">93.0% <span style="font-size:13px;background:#eafaf3;color:#168452;padding:7px 12px;border-radius:20px">APPROVED</span></h1><div class="factor">Credit Score 750</div><div class="factor">Annual Income ₹90,000</div><div class="factor">Loan Amount ₹25,000</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Model at a glance</div>',unsafe_allow_html=True); st.write('')
    for c,(n,l) in zip(st.columns(4),[('85.11%','Test Accuracy'),('82.03%','ROC-AUC'),('52,000','Applications'),('25','Input Features')]):
        c.markdown(f'<div class="metric"><div class="num">{n}</div><div class="label">{l}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">How it works</div>',unsafe_allow_html=True); st.write('')
    steps=[('01','Enter applicant details'),('02','Preprocess inputs'),('03','Run Random Forest'),('04','Review assessment')]
    for c,(n,t) in zip(st.columns(4),steps):
        c.markdown(f'<div class="card"><div class="blue"><b>{n}</b></div><h4 style="color:#17394d">{t}</h4><div class="muted">Complete one step of the end-to-end ML pipeline.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Top model factors</div>',unsafe_allow_html=True)
    for c,(n,v) in zip(st.columns(5),[('Credit Score','21.18%'),('Loan Amount','18.31%'),('Annual Income','11.03%'),('Age','6.95%'),('Interest Rate','3.82%')]):
        c.markdown(f'<div class="factor"><div class="muted">{n}</div><div class="num">{v}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:12px;"><div style="color:#5d7482;line-height:1.6;font-size:14px;"><b style="color:#17394d;">Credit Score is the most important feature</b> in the trained Random Forest model at approximately 21.18%. Feature importance indicates model reliance, not causation.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="footer">LoanLens AI • Machine Learning-II Project • Decision-support prototype</div>',unsafe_allow_html=True)

def assessment():
    nav(); st.markdown('<div class="section">Loan Assessment</div><div class="sub">Enter applicant information to generate a model-based prediction.</div>',unsafe_allow_html=True)
    with st.form('loan'):
        c1,c2,c3=st.columns(3)
        gender=c1.selectbox('Gender',['Male','Female']); age=c1.number_input('Age',18,69,35); marital=c1.selectbox('Marital Status',['Married','Single','Divorced']); dependents=c1.number_input('Dependents',0,3,2)
        education=c2.selectbox('Education',['Graduate','High School','Postgraduate']); employment=c2.selectbox('Employment Status',['Employed','Self-Employed','Unemployed']); occupation=c2.selectbox('Occupation Type',['Salaried','Business','Freelancer','Professional']); residential=c2.selectbox('Residential Status',['Own','Rent','Other'])
        city=c3.selectbox('City/Town',['Urban','Suburban','Rural']); annual_income=c3.number_input('Annual Income (₹)',20009,149998,90000); monthly_expenses=c3.number_input('Monthly Expenses (₹)',500,4999,3000); credit_score=c3.number_input('Credit Score',300,849,750)
        c1,c2,c3=st.columns(3)
        existing_loans=c1.number_input('Existing Loans',0,2,1); existing_loan_amount=c1.number_input('Total Existing Loan Amount (₹)',0,49999,20000); outstanding_debt=c1.number_input('Outstanding Debt (₹)',0,29998,10000); loan_history=c1.selectbox('Loan History',[1,0])
        loan_amount=c2.number_input('Loan Amount Requested (₹)',5000,44848,25000); loan_term=c2.number_input('Loan Term (Months)',12,239,120); loan_purpose=c2.selectbox('Loan Purpose',['Home','Vehicle','Personal','Education']); interest_rate=c2.number_input('Interest Rate (%)',3.5,15.0,7.5,step=0.1)
        loan_type=c3.selectbox('Loan Type',['Secured','Unsecured']); co_applicant=c3.selectbox('Co-Applicant',['Yes','No']); bank_history=c3.number_input('Bank Account History',0,9,7); transaction_frequency=c3.number_input('Transaction Frequency',5,29,20); default_risk=c3.number_input('Default Risk',0.0,1.0,0.15,step=0.01)
        go=st.form_submit_button('Generate Loan Assessment →',type='primary',use_container_width=True)
    if go:
        d=pd.DataFrame({'Gender':[gender],'Age':[age],'Marital_Status':[marital],'Dependents':[dependents],'Education':[education],'Employment_Status':[employment],'Occupation_Type':[occupation],'Residential_Status':[residential],'City/Town':[city],'Annual_Income':[annual_income],'Monthly_Expenses':[monthly_expenses],'Credit_Score':[credit_score],'Existing_Loans':[existing_loans],'Total_Existing_Loan_Amount':[existing_loan_amount],'Outstanding_Debt':[outstanding_debt],'Loan_History':[loan_history],'Loan_Amount_Requested':[loan_amount],'Loan_Term':[loan_term],'Loan_Purpose':[loan_purpose],'Interest_Rate':[interest_rate],'Loan_Type':[loan_type],'Co-Applicant':[co_applicant],'Bank_Account_History':[bank_history],'Transaction_Frequency':[transaction_frequency],'Default_Risk':[default_risk]})
        pred=int(model.predict(d)[0]); p=float(model.predict_proba(d)[0][1]*100); status='Approved' if pred==1 else 'Rejected'
        st.session_state.last={**d.iloc[0].to_dict(),'status':status,'probability':p,'explanation':explain(d.iloc[0])}
    a=st.session_state.last
    if a:
        cls='approved' if a['status']=='Approved' else 'rejected'
        st.markdown(f'<div class="{cls}"><div class="result">{a["status"]}</div><div class="muted">Model-based loan approval classification</div><div class="prob">{a["probability"]:.2f}%</div><div class="muted">Approval probability</div></div>',unsafe_allow_html=True)
        c1,c2=st.columns([1.2,.8])
        with c1:
            st.markdown('<div class="card"><h3 style="color:#17394d">Why the model reached this result</h3>',unsafe_allow_html=True)
            for z in a['explanation']: st.markdown('• '+z)
            st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card"><h3 style="color:#17394d">Key Inputs</h3><p class="muted">Credit Score</p><h2 class="blue">{a["Credit_Score"]}</h2><p class="muted">Annual Income</p><h3 style="color:#12679b">₹{a["Annual_Income"]:,.0f}</h3><p class="muted">Loan Amount</p><h3 style="color:#12679b">₹{a["Loan_Amount_Requested"]:,.0f}</h3></div>',unsafe_allow_html=True)
        st.download_button('⬇ Download Assessment PDF',pdf(a),'LoanLens_AI_Assessment.pdf','application/pdf',use_container_width=True)

def performance():
    nav(); st.markdown('<div class="section">Model Performance</div><div class="sub">Comparison of Logistic Regression and Random Forest on the project test set.</div>',unsafe_allow_html=True); st.write('')
    comparison=[('Accuracy',85.07,85.11),('Precision',85.03,85.10),('Recall',93.12,93.09),('F1-Score',88.89,88.91),('ROC-AUC',81.52,82.03)]
    rows_html=''.join([f'<tr><td class="metric-name">{m}</td><td>{lr:.2f}%</td><td><span class="rf-value">{rf:.2f}%</span></td></tr>' for m,lr,rf in comparison])
    st.markdown(f'<div class="perf-card"><div class="perf-card-head"><div><div class="perf-title">Model comparison</div><div class="perf-subtitle">Evaluation on the project test set</div></div><div class="model-pill">Random Forest</div></div><div class="comparison-table-wrap"><table class="comparison-table"><thead><tr><th>Metric</th><th>Logistic Regression</th><th>Random Forest</th></tr></thead><tbody>{rows_html}</tbody></table></div><div class="table-note">Random Forest is the deployed model based on its slightly higher overall test-set performance.</div></div>',unsafe_allow_html=True)
    kpis=[('85.11%','Random Forest Accuracy','Test set'),('82.03%','Random Forest ROC-AUC','Test set'),('88.91%','Random Forest F1-Score','Test set'),('52,000','Dataset Applications','Total rows')]
    for col,(value,title,note) in zip(st.columns(4),kpis): col.markdown(f'<div class="metric"><div class="num">{value}</div><div class="label" style="font-size:14px;margin-top:4px;">{title}</div><div style="color:#9aabb5;font-size:11px;margin-top:5px;">{note}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Performance by metric</div><div class="sub">Visual comparison of the same test-set metrics.</div>',unsafe_allow_html=True); st.write('')
    bars_html=''.join([f'<div class="perf-bar-row"><div class="perf-bar-label">{m}</div><div class="bar-line"><span class="bar-model">LR</span><div class="bar-track"><div class="bar-fill lr-fill" style="width:{lr}%;"></div></div><span class="bar-number">{lr:.2f}%</span></div><div class="bar-line"><span class="bar-model">RF</span><div class="bar-track"><div class="bar-fill rf-fill" style="width:{rf}%;"></div></div><span class="bar-number">{rf:.2f}%</span></div></div>' for m,lr,rf in comparison])
    st.markdown(f'<div class="perf-card"><div class="legend-row"><span><span class="legend-dot"></span>Logistic Regression</span><span><span class="legend-dot rf-dot"></span>Random Forest</span></div>{bars_html}</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Confusion matrices</div><div class="sub">Evaluation on test set predictions (10,400 samples).</div>',unsafe_allow_html=True); st.write('')
    cm_html='''<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:24px;"><div class="cm-card"><div class="cm-title">Logistic Regression</div><div class="cm-subtitle">Confusion Matrix (Test Set)</div><div class="cm-grid"><div></div><div class="cm-label">Pred 0</div><div class="cm-label">Pred 1</div><div class="cm-label side">Actual 0</div><div class="cm-cell tn"><small>True Neg</small><strong>2,633</strong></div><div class="cm-cell fp"><small>False Pos</small><strong>1,094</strong></div><div class="cm-label side">Actual 1</div><div class="cm-cell fn"><small>False Neg</small><strong>459</strong></div><div class="cm-cell tp"><small>True Pos</small><strong>6,214</strong></div></div></div><div class="cm-card"><div class="cm-title">Random Forest</div><div class="cm-subtitle">Confusion Matrix (Test Set)</div><div class="cm-grid"><div></div><div class="cm-label">Pred 0</div><div class="cm-label">Pred 1</div><div class="cm-label side">Actual 0</div><div class="cm-cell tn"><small>True Neg</small><strong>2,639</strong></div><div class="cm-cell fp"><small>False Pos</small><strong>1,088</strong></div><div class="cm-label side">Actual 1</div><div class="cm-cell fn"><small>False Neg</small><strong>461</strong></div><div class="cm-cell tp"><small>True Pos</small><strong>6,212</strong></div></div></div></div>'''
    st.markdown(cm_html,unsafe_allow_html=True)
    st.markdown('<div class="section">Random Forest feature importance</div><div class="sub">Top ten impurity-based feature importance values from the trained Random Forest.</div>',unsafe_allow_html=True); st.write('')
    features=[('Credit Score',21.1834),('Loan Amount Requested',18.3090),('Annual Income',11.0275),('Age',6.9462),('Interest Rate',3.8224),('Outstanding Debt',3.7987),('Monthly Expenses',3.7705),('Total Existing Loan Amount',3.7484),('Loan Term',3.5441),('Default Risk',3.3607)]
    max_importance=features[0][1]
    feature_rows=''.join([f'<div class="fi-row"><div class="fi-name">{name}</div><div class="fi-track"><div class="fi-fill" style="width:{(value/max_importance)*100:.2f}%;"></div></div><div class="fi-value">{value:.2f}%</div></div>' for name,value in features])
    st.markdown(f'<div class="perf-card"><div class="perf-card-head"><div><div class="perf-title">What the model relied on most</div><div class="perf-subtitle">Relative importance within the trained Random Forest</div></div><div class="importance-badge">Top 10</div></div>{feature_rows}<div class="table-note" style="margin-top:18px;">Credit Score is the most important feature at approximately 21.18%. Feature importance indicates model reliance, not causation.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:20px;"><div style="font-size:20px;font-weight:850;color:#17394d;margin-bottom:8px;">Model selection</div><div class="muted" style="line-height:1.7;font-size:14px;">Random Forest is deployed because it achieved slightly higher accuracy, precision, F1-score, and ROC-AUC than Logistic Regression on the project test set. Logistic Regression achieved a marginally higher recall. These results are specific to this dataset and should not be interpreted as bank-grade or real-world lending performance.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="footer">LoanLens AI • Model Evaluation Dashboard</div>',unsafe_allow_html=True)

if st.session_state.page=='home': home()
elif st.session_state.page=='assessment': assessment()
elif st.session_state.page=='performance': performance()
