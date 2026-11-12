from pathlib import Path
import json, re, math, statistics, hashlib, csv, shutil
from copy import deepcopy
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT=Path(__file__).resolve().parents[1]; ROOT=Path('/Users/mauryavardhansingh/Desktop/BTP/BTP')
E=json.loads((OUT/'_work/evidence.json').read_text())
def rows(f):return E[f]['rows']
F=rows('results/prof_tasks/FINAL_results.csv')
A=[r for r in rows('results/prof_tasks/audit_v2_features.csv') if r['set']=='F0_base5']
C=rows('results/prof_tasks/audit_ood_conformal_strict.csv')
IDS=['D1_NRC','D2_Zhao2020','D3_KAERI_uniform','D4_KAERI_nonuniform','D5_Helical_R123','D6_Hardik_helical_water','D7_Hardik_straight_R123','D8_Pioro_R134a']
LABELS=['NRC','Zhao','KAERI uniform','KAERI nonuniform','Helical R-123','Hardik water','Straight R-123','Pioro R-134a']
NAMES=['Mauryavardhan Singh','Mayank Nagar','Ayushman Paul','Hardik Mahawar','Bikash Pattanayak']
EMAILS=['singhmauryavardhan@gmail.com','nagarmayank2004@gmail.com','mr.infosagepoint2020@gmail.com','mahawar.hardik0107@gmail.com','bikash.ptnk@gmail.com']
TITLE='Physics anchored machine learning for critical heat flux prediction under dataset shift'
AFFIL='Indian Institute of Technology Kharagpur, Kharagpur, West Bengal 721302, India'

REFERENCES=[
'Bell, I.H., Wronski, J., Quoilin, S., Lemort, V., 2014. Pure and pseudo-pure fluid thermophysical property evaluation and the open-source thermophysical property library CoolProp. Industrial & Engineering Chemistry Research 53, 2498–2508. https://doi.org/10.1021/ie4033999.',
'Bourisaw, R., McCants, R., Le Corre, J.-M., Iskhakova, A., Iskhakov, A., 2026. Critical Heat Flux Datasets under Uniform and Non-Uniform Axial Heating Conditions [KAERI/TR-1665/2000]. Zenodo, version 2. https://doi.org/10.5281/zenodo.18404759.',
'Geurts, P., Ernst, D., Wehenkel, L., 2006. Extremely randomized trees. Machine Learning 63, 3–42. https://doi.org/10.1007/s10994-006-6226-1.',
'Groeneveld, D.C., Shan, J.Q., Vasić, A.Z., Leung, L.K.H., Durmayaz, A., Yang, J., Cheng, S.C., Tanase, A., 2007. The 2006 CHF look-up table. Nuclear Engineering and Design 237, 1909–1922. https://doi.org/10.1016/j.nucengdes.2007.02.014.',
'Hardik, B.K., Kumar, G., Prabhu, S.V., 2017. Boiling pressure drop, local heat transfer distribution and critical heat flux in horizontal straight tubes. International Journal of Heat and Mass Transfer 113, 466–481. https://doi.org/10.1016/j.ijheatmasstransfer.2017.05.073.',
'Hardik, B.K., Prabhu, S.V., 2017. Critical heat flux in helical coils at low pressure. Applied Thermal Engineering 112, 1223–1239. https://doi.org/10.1016/j.applthermaleng.2016.10.114.',
'Hardik, B.K., Prabhu, S.V., 2018. Boiling pressure drop, local heat transfer distribution and critical heat flux in helical coils with R123. International Journal of Thermal Sciences 125, 149–165. https://doi.org/10.1016/j.ijthermalsci.2017.11.026.',
'Kapoor, S., Narayanan, A., 2023. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 4, 100804. https://doi.org/10.1016/j.patter.2023.100804.',
'Katto, Y., Ohno, H., 1984. An improved version of the generalized correlation of critical heat flux for the forced convective boiling in uniformly heated vertical tubes. International Journal of Heat and Mass Transfer 27, 1641–1648. https://doi.org/10.1016/0017-9310(84)90276-X.',
'Le Corre, J.-M., Delipei, G., Wu, X., Zhao, X., 2024. Benchmark on Artificial Intelligence and Machine Learning for Scientific Computing in Nuclear Engineering. Phase 1: Critical Heat Flux Exercise Specifications. NEA/WKP(2023)1, OECD Nuclear Energy Agency, January 2024.',
'Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R.J., Wasserman, L., 2018. Distribution-free predictive inference for regression. Journal of the American Statistical Association 113, 1094–1111. https://doi.org/10.1080/01621459.2017.1307116.',
'Mondal, A., Sharma, S.L., 2025. Bayesian neural network for critical heat flux prediction: a unified approach with uncertainty quantification. Applied Thermal Engineering 278, 127418. https://doi.org/10.1016/j.applthermaleng.2025.127418.',
'Park, C., 2000. Effect of Axial Heat Flux Distribution on Critical Heat Flux. Technical Report KAERI/TR-1665/2000, Korea Atomic Energy Research Institute, Daejeon, Republic of Korea.',
'Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É., 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html.',
'Pioro, I.L., Groeneveld, D.C., Leung, L.K.H., Doerffer, S.S., Cheng, S.C., Antoshko, Yu.V., Guo, Y., Vasić, A., 2002. Comparison of CHF measurements in horizontal and vertical tubes cooled with R-134a. International Journal of Heat and Mass Transfer 45, 4435–4450. https://doi.org/10.1016/S0017-9310(02)00149-7.',
'Saraswat, S.P., 2026. Campaign-disjoint validation of machine-learned critical heat flux models: leakage quantification and physics-anchored residual transfer. Energy and AI 26, 100889. https://doi.org/10.1016/j.egyai.2026.100889.',
'Yang, S., Li, X., Ren, B., Yang, L., Chen, C., Lu, Q., Wei, Z., 2026. A transfer learning framework for critical heat flux prediction pretrained on the 2006 lookup table. Applied Thermal Engineering 287, 129451. https://doi.org/10.1016/j.applthermaleng.2025.129451.',
'Zhao, X., Shirvan, K., Salko, R.K., Guo, F., 2020. On the prediction of critical heat flux using a physics-informed machine learning-aided framework. Applied Thermal Engineering 164, 114540. https://doi.org/10.1016/j.applthermaleng.2019.114540.'
]

CAPTIONS={
1:'Figure 1. Record counts in the eight-dataset flow corpus. The horizontal axis is logarithmic. Colours distinguish water and refrigerant datasets; counts refer to the archived experimental populations after the stated preprocessing, not to new measurements.',
2:'Figure 2. Evaluated prediction pathway. Local quality can enter the archived physics arms and their disagreement even though it is not a direct input to the residual learner. The calibration slice and heuristic correction bound are shown separately. This schematic does not imply equivalence with the later inlet-only query interface.',
3:'Figure 3. Mean fractions within ±10% of observed CHF under random 80/20, grouped and leave-one-dataset-out evaluation. Random values summarize ten seeds, grouped values summarize folds with seed 0, and dataset-holdout values summarize five seeds. Grouped evaluation is unavailable for straight R-123. Group meanings vary by dataset.',
4:'Figure 4. Five-seed leave-one-dataset-out performance. Panel (a) shows mean R² with one standard deviation across seeds; panel (b) shows the mean fraction within ±10%. These error bars are not confidence intervals and do not incorporate source-selection or measurement uncertainty.',
5:'Figure 5. Baseline and anchored variants in the separate three-seed base-feature comparison. The unbounded and bounded predictions use the same fitted residual model. The selected baseline is Katto in all eight folds. Direct-learning values are reported in Table 5 rather than omitted from the numerical comparison.',
6:'Figure 6. Empirical coverage in the separate seed-0 interval audit. Pairwise source exclusion excludes both the target dataset and the residual-generating dataset during calibration-model fitting. The additional widening factor is 1 + (1 − t). The dashed line marks the nominal 90% level, not a proven coverage guarantee.'
}

def setfont(run,size=None,bold=None):
 run.font.name='Times New Roman';run.font.color.rgb=RGBColor(0,0,0)
 if size:run.font.size=Pt(size)
 if bold is not None:run.bold=bold
 rp=run._element.get_or_add_rPr();rf=rp.find(qn('w:rFonts'))
 if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
 for a in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+a),'Times New Roman')
 for a in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
  if qn('w:'+a) in rf.attrib:del rf.attrib[qn('w:'+a)]

def setup(title):
 d=Document();s=d.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11)
 s.top_margin=s.bottom_margin=Inches(.8);s.left_margin=s.right_margin=Inches(1)
 s.header_distance=s.footer_distance=Inches(.35)
 for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption','List Bullet']:
  st=d.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
  st.font.size=Pt(12 if name=='Normal' else 11)
  st.paragraph_format.space_after=Pt(6);st.paragraph_format.line_spacing=1.15
  rp=st.element.get_or_add_rPr();rf=rp.find(qn('w:rFonts'))
  if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
  for a in ['ascii','hAnsi','cs','eastAsia']:rf.set(qn('w:'+a),'Times New Roman')
  for a in list(rf.attrib):
   if 'Theme' in a or 'theme' in a:del rf.attrib[a]
 d.styles['Title'].font.size=Pt(17);d.styles['Title'].font.bold=True
 d.styles['Title'].paragraph_format.space_after=Pt(12)
 for name,size in [('Heading 1',13),('Heading 2',12)]:
  st=d.styles[name];st.font.size=Pt(size);st.font.bold=True;st.paragraph_format.space_before=Pt(12);st.paragraph_format.keep_with_next=True
 d.styles['Caption'].font.size=Pt(10);d.styles['Caption'].font.italic=False;d.styles['Caption'].paragraph_format.line_spacing=1.05
 d.styles['Caption'].font.bold=False
 for st in d.styles:
  for border in st.element.findall('.//' + qn('w:pBdr')):
   border.getparent().remove(border)
 d.core_properties.title=title;d.core_properties.author='; '.join(NAMES);d.core_properties.subject='Retrospective CHF prediction and source transfer';d.core_properties.keywords='CHF, residual learning, dataset shift'
 p=s.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 r=p.add_run();fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');r._r.addnext(fld);setfont(r,10)
 return d

def para(d,text,style=None):
 p=d.add_paragraph(style=style)
 pattern=r'(Δh_sub|h_fg|ρ_l|ρ_g|P_c|u_d|u_s|u_f|q_0\.9|y_i|ŷ_i)'
 for part in re.split(pattern,text):
  if re.fullmatch(pattern,part):
   base,index=part.split('_');setfont(p.add_run(base));r=p.add_run(index);setfont(r);r.font.subscript=True
  else:setfont(p.add_run(part))
 p.paragraph_format.widow_control=True
 return p

def table(d,caption,headers,data,widths,note=''):
 p=para(d,caption,'Caption');p.paragraph_format.keep_with_next=True
 t=d.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for i,w in enumerate(widths):t.columns[i].width=Inches(w)
 for i,h in enumerate(headers):t.rows[0].cells[i].text=h
 for row in data:
  for cell,value in zip(t.add_row().cells,row):cell.text=str(value)
 pr=t._tbl.tblPr;b=OxmlElement('w:tblBorders')
 for edge in ['top','left','bottom','right','insideH','insideV']:
  x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');b.append(x)
 pr.append(b)
 for ri,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
  if ri==0:trpr.append(OxmlElement('w:tblHeader'))
  for ci,cell in enumerate(row.cells):
   cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   cp=cell._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
   for edge,val in [('top',70),('bottom',70),('left',80),('right',80)]:
    el=OxmlElement('w:'+edge);el.set(qn('w:w'),str(val));el.set(qn('w:type'),'dxa');mar.append(el)
   cp.append(mar)
   if ri==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EFEFEF');cp.append(sh)
   for p in cell.paragraphs:
    p.paragraph_format.keep_with_next=ri<len(t.rows)-1
    p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.03
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:setfont(r,10,ri==0)
 if note:
  p=para(d,note,'Caption');p.paragraph_format.space_before=Pt(4)
 else:para(d,'').paragraph_format.space_after=Pt(0)
 return t

def figure(d,n):
 p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(3)
 shape=p.add_run().add_picture(str(OUT/'figures'/f'Figure_{n}.png'),width=Inches(6.5))
 shape._inline.docPr.set('descr',CAPTIONS[n]);shape._inline.docPr.set('title',f'Figure {n}')
 para(d,CAPTIONS[n],'Caption')

# Editable native Word math constructors.
def mr(s):
 e=OxmlElement('m:r');pr=OxmlElement('m:rPr');nor=OxmlElement('m:nor');pr.append(nor);e.append(pr)
 wr=OxmlElement('w:rPr');fs=OxmlElement('w:rFonts');fs.set(qn('w:ascii'),'Times New Roman');fs.set(qn('w:hAnsi'),'Times New Roman');wr.append(fs);e.append(wr)
 t=OxmlElement('m:t');t.text=s;t.set(qn('xml:space'),'preserve');e.append(t);return e
def seq(v):return [mr(v)] if isinstance(v,str) else v
def fraction(a,b):
 f=OxmlElement('m:f')
 for tag,v in [('num',a),('den',b)]:
  x=OxmlElement('m:'+tag)
  for z in seq(v):x.append(z)
  f.append(x)
 return f
def sub(a,b):
 s=OxmlElement('m:sSub')
 for tag,v in [('e',a),('sub',b)]:
  e=OxmlElement('m:'+tag)
  for z in seq(v):e.append(z)
  s.append(e)
 return s
def sup(a,b):
 s=OxmlElement('m:sSup')
 for tag,v in [('e',a),('sup',b)]:
  e=OxmlElement('m:'+tag)
  for z in seq(v):e.append(z)
  s.append(e)
 return s
def eq(d,n):
 if d.paragraphs:d.paragraphs[-1].paragraph_format.keep_with_next=True
 p=d.add_paragraph();p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(10)
 p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5),WD_TAB_ALIGNMENT.RIGHT)
 om=OxmlElement('m:oMath')
 if n==1:v=[sub('x','out'),mr(' = −'),fraction([sub('Δh','sub')],[sub('h','fg')]),mr(' + '),fraction('4Lq″',[mr('GD'),sub('h','fg')])]
 elif n==2:v=[mr('φ = ['),fraction([mr('σ'),sub('ρ','l')],[sup('G','2'),mr('L')]),mr(', '),fraction([sub('ρ','g')],[sub('ρ','l')]),mr(', '),fraction('L','D'),mr(', '),fraction([sub('Δh','sub')],[sub('h','fg')]),mr(', '),fraction('P',[sub('P','c')]),mr(']')]
 elif n==3:v=[sub('r','i'),mr(' = ln('),fraction([sub('y','i')],[sub('B','i')]),mr('),     r̂ = f(φ)')]
 elif n==4:v=[mr('ŷ = B exp{clip[r̂, −c(t), c(t)]}')]
 elif n==5:v=[mr('c(t) = ln(1.5) + t[ln(10) − ln(1.5)]')]
 elif n==6:v=[mr('t = 1 − max('),sub('u','d'),mr(', '),sub('u','s'),mr(', '),sub('u','f'),mr(')')]
 elif n==7:v=[mr('I = [ŷ exp(−w), ŷ exp(w)],    w = '),sub('q','0.9'),mr('[1 + 2(1 − t)]')]
 elif n==8:v=[sup('R','2'),mr(' = 1 − '),fraction([mr('Σ'),sup('(yᵢ − ŷᵢ)','2')],[mr('Σ'),sup('(yᵢ − ȳ)','2')])]
 elif n==9:v=[mr('MAPE = '),fraction('100','n'),mr('Σ |'),fraction('ŷᵢ − yᵢ','yᵢ'),mr('|,    '),sub('W','10'),mr(' = '),fraction('1','n'),mr('Σ 1{'),fraction('|ŷᵢ − yᵢ|','yᵢ'),mr(' ≤ 0.10}')]
 for z in v:om.append(z)
 p._p.append(om);setfont(p.add_run('\t('+str(n)+')'),11)

def getf(d,kind):return next((r for r in F if r['dataset']==d and r['split'].startswith(kind)),None)
def fmt(x,p=3):return f'{x:.{p}f}' if x is not None and math.isfinite(x) else '—'
def main_table(d,n):
 if n==1:
  data=[['NRC','Water','24,443','Reference group (60)'],['Zhao','Water','1,864','Author group (10)'],['KAERI uniform','Water','651','Source group (10)'],['KAERI nonuniform','Water','888','Source group (11)'],['Helical R-123','R-123','257','Pressure band (2)'],['Hardik water','Water','156','Coil identity (6)'],['Straight R-123','R-123','55','Not available'],['Pioro R-134a','R-134a','268','Figure group (9)'],['Total','','28,582','']]
  table(d,'Table 1. Principal flow corpus and group definitions.',['Dataset','Fluid','Records','Grouping variable'],data,[1.6,.8,.7,3.4], 'Counts are those used by the flow evaluation. Pioro values are digitized. Original source overlap and independence are not established by these counts.')
 elif n==2:
  data=[['Within dataset','80/20 and 70/30','10 seeds','Duplicate-signature groups'],['Grouped','Up to 10 folds','Seed 0','Dataset-specific groups'],['Dataset holdout','Eight held-out datasets','5 seeds','Complete source file'],['Anchoring ablation','Eight held-out datasets','3 seeds','Complete source file'],['Interval audit','Pairwise exclusion plus LODO','Seed 0','Target excluded from calibration fits']]
  table(d,'Table 2. Distinct evaluation families.',['Evaluation','Partition','Repetition','Unit withheld'],data,[1.3,1.65,.85,2.7],'LODO denotes leave one dataset out. Grouped standard deviations summarize folds; they are not multi-seed estimates.')
 elif n==3:
  data=[]
  for did,lab in zip(IDS,LABELS):
   r,g,l=getf(did,'80/20'),getf(did,'GROUPED'),getf(did,'LODO')
   data.append([lab,fmt(r['R2_mean']),fmt(g['R2_mean']) if g else '—',fmt(l['R2_mean'])])
  table(d,'Table 3. R² under three prediction tasks.',['Dataset','Random 80/20','Grouped mean','Dataset holdout'],data,[2.0,1.5,1.5,1.5],'The three columns differ in partition and training population. Grouped values average folds, and dataset-holdout values average five seeds. Supplementary tables include 70/30 results.')
 elif n==4:
  data=[]
  for did,lab in zip(IDS,LABELS):
   r=getf(did,'LODO');data.append([lab,f"{r['R2_mean']:.3f} ± {r['R2_sd']:.3f}",f"{100*r['within_10pct_mean']:.1f}",f"{r['MAPE_pct_mean']:.1f}",f"{100*r['coverage90_mean']:.1f}"])
  vals=[getf(did,'LODO') for did in IDS]
  data.append(['Across-dataset median',fmt(statistics.median(r['R2_mean'] for r in vals)),f"{100*statistics.median(r['within_10pct_mean'] for r in vals):.1f}",f"{statistics.median(r['MAPE_pct_mean'] for r in vals):.1f}",f"{100*statistics.median(r['coverage90_mean'] for r in vals):.1f}"])
  table(d,'Table 4. Five-seed leave-one-dataset-out summary.',['Dataset','R² mean ± SD','Within ±10% (%)','MAPE (%)','Coverage (%)'],data,[1.65,1.55,1.1,.85,1.35],'Coverage refers to intervals with nominal 90% coverage. SD is across five seeds. The median is unweighted across datasets; no SD is assigned to that summary.')
 elif n==5:
  arms=['backbone_alone','pooled_ML','PIPE_unbounded','PIPE'];data=[]
  for did,lab in zip(IDS,LABELS):data.append([lab]+[fmt(next(r for r in A if r['held_out']==did and r['arm']==a)['R2']) for a in arms])
  data.append(['Median']+[fmt(statistics.median(r['R2'] for r in A if r['arm']==a)) for a in arms])
  table(d,'Table 5. R² in the separate three-seed base-feature comparison.',['Held-out dataset','Baseline','Direct ML','Unbounded residual','Bounded residual'],data,[1.7,1,1.1,1.35,1.35],'All methods use the same outer dataset partitions and five numerical inputs. Direct ML has a larger fitting population because it does not reserve the residual-calibration slice. Bounds are the sole prediction difference between the two residual columns.')
 elif n==6:
  cols=['cov_shipped','cov_strict','cov_strict_trust'];data=[]
  for did,lab in zip(IDS,LABELS):
   r=next(r for r in C if r['held_out']==did);data.append([lab]+[f'{100*r[k]:.1f}' for k in cols])
  data.append(['Median']+[f'{100*statistics.median(r[k] for r in C):.1f}' for k in cols])
  data.append(['Minimum']+[f'{100*min(r[k] for r in C):.1f}' for k in cols])
  table(d,'Table 6. Prediction-interval coverage in the separate seed-0 audit.',['Held-out dataset','Default (%)','Pairwise exclusion (%)','Exclusion plus widening (%)'],data,[1.8,1.2,1.6,1.9],'These are empirical coverages, not guaranteed probabilities. They must not be substituted for the five-seed coverage values in Table 4.')

def render_text(d,source):
 source=re.sub(r'(?m)^(#{1,2} .+)$',r'\n\1\n',source)
 for block in re.split(r'\n\s*\n',source.strip()):
  block=block.strip()
  if block.startswith('## '):d.add_heading(block[3:],level=2)
  elif block.startswith('# '):d.add_heading(block[2:],level=1)
  elif re.fullmatch(r'\{\{TABLE\d\}\}',block):main_table(d,int(re.search(r'\d',block).group()))
  elif re.fullmatch(r'\{\{FIG\d\}\}',block):figure(d,int(re.search(r'\d',block).group()))
  elif re.fullmatch(r'\{\{EQ\d\}\}',block):eq(d,int(re.search(r'\d',block).group()))
  elif block=='{{REFERENCES}}':
   for ref in REFERENCES:
    p=para(d,ref);p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(6)
    p.paragraph_format.left_indent=Inches(.2);p.paragraph_format.first_line_indent=Inches(-.2)
    for r in p.runs:setfont(r,10.5)
  else:para(d,block)

doc=setup(TITLE);para(doc,TITLE,'Title')
p=para(doc,', '.join(NAMES));p.paragraph_format.space_after=Pt(6)
para(doc,AFFIL)
for name,email in zip(NAMES,EMAILS):
 p=para(doc,f'{name}: {email}');p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1
 for r in p.runs:setfont(r,10)
render_text(doc,(OUT/'_work/paper_text.md').read_text())
doc.save(OUT/'CHF_revised_manuscript.docx')

highlights=[
'Eight flow datasets reveal a gap between random and source-holdout CHF prediction.',
'Physics anchoring improves transfer more consistently than the correction bound.',
'Median dataset-holdout R² is 0.711, with 31.8% within ±10% of measured CHF.',
'Nominal 90% intervals achieve only 53.2% median dataset-holdout coverage.',
'Quality and metadata limitations prevent a universal deployment claim.'
]
assert all(len(t)<=85 for t in highlights),[(len(t),t) for t in highlights]
h=setup('Highlights');para(h,'Highlights','Title');para(h,TITLE)
for t in highlights:para(h,t,'List Bullet')
h.save(OUT/'Highlights.docx')

s=setup('Supporting benchmarks and experiment provenance');para(s,'Supporting benchmarks and experiment provenance','Title');para(s,TITLE)
s.add_heading('S1 Interpretation of the supporting experiments',1)
para(s,'The main manuscript concerns the 28,582-row flow corpus. The earlier unified benchmark contains 28,470 rows from seven sources: NRC 24,579; Zhao 1,865; KAERI uniform 651; KAERI nonuniform 888; helical R-123 257; pin-fin water/FC-72 175; and a pool-boiling strip dataset with 55 rows. These populations overlap substantially with the flow corpus but are not the same experiment. Supporting results are not used to extend the scope of the final flow model to pool boiling or engineered surfaces.')
para(s,'The unified master table has 51 columns. Model inputs comprise seven core variables, seven property-related variables, six geometry indicators and three fluid indicators. The target is a log ratio to a physical scale: a flow scale based on G h_fg or a pool-boiling scale based on Zuber-type normalization. Accordingly, the models in Table S1 are not physics-free baselines. Median imputation and other model-specific missing-data handling are used. Their experimental CHF range is approximately 37.7–28,800 kW m⁻², distinct from the positive lookup-table range.')
para(s,'Strategy 1 is a source-stratified random split with 22,776 training and 5,694 test rows, using seed 42. Strategy 2 withholds each eligible source’s upper-pressure range and has 6,619 test rows; quantile ties mean this is not an exact 20% split. Sources without the relevant pressure variation remain in training. Strategy 3 holds out the 175 pin-fin and 257 helical rows jointly. It therefore changes source, geometry, fluid and regime rather than isolating a surface effect. Strategy 4 holds out each source file and pools the out-of-fold predictions. Its pooled R² is not a median of source-wise R².')
U=rows('unified_chf_pipeline/results/metrics_summary.csv')
models=['HistGB','RandomForest','ANN','GPR','PINN'];tags=['strategy1','strategy2','strategy3','strategy4_pooled_oof']
data=[[m]+[fmt(next(r for r in U if r['model']==m and r['split_tag']==tag)['R2']) for tag in tags] for m in models]
table(s,'Table S1. R² in the earlier seven-source benchmark.',['Model','Strategy 1','Strategy 2','Strategy 3','Strategy 4 pooled'],data,[1.4,1.2,1.2,1.2,1.5],'Values were recalculated from stored row predictions during manuscript verification. Training was not rerun. Extremely negative values are retained because they occur in the recorded predictions.')
para(s,'The ANN uses median imputation with indicators, scaling, hidden layers of 64 and 32 units, early stopping and a maximum of 500 iterations. Random Forest uses 500 trees with minimum leaf size 2. HistGB uses 500 iterations, learning rate 0.05 and at most 63 leaves. GPR uses imputed and scaled inputs, an RBF-based kernel with a noise component and at most 800 fitting rows. The merged-data PINN uses a 32/16 SiLU network, a bounded output, Adam training, early stopping and a grid of penalty weights 0, 0.05 and 0.2. It is distinct from the earlier LUT PINN.')
data=[]
for m in models:
 r=next(r for r in U if r['model']==m and r['split_tag']=='strategy1')
 data.append([m,fmt(r['R2']),fmt(r['RMSE'],1),fmt(r['MAE'],1),fmt(r['MAPE_pct'],1),fmt(r['train_seconds'],1)])
table(s,'Table S2. Strategy 1 metrics from the earlier seven-source benchmark.',['Model','R²','RMSE','MAE','MAPE (%)','Fit time (s)'],data,[1.4,.75,1.05,1.05,1.1,1.15],'RMSE and MAE are in kW m⁻². Timing is the recorded fit duration, not matched end-to-end cost. The GPR fitting cap and absent common hardware record prevent a fair general computational-efficiency conclusion.')
s.add_heading('S2 Earlier physics ablation',1)
para(s,'The physics ablation uses the same seven-source, 28,470-row population and stored split strategies, not the eight-dataset flow corpus. It combines revised property handling, physical scales and alternative feature or correction formulations. The labels below identify those configurations; they do not isolate the causal effect of each preprocessing repair.')
P=rows('physics_pipeline/results/ablation_metrics.csv');arms=['PHYS_katto','A0_no_physics','A1_latent_baseline','A2_katto_baseline','A3_pi_features','A5_constrained']
table(s,'Table S3. R² in the earlier physics ablation.',['Variant','Strategy 1','Strategy 2','Strategy 3','Strategy 4 pooled'],[[a]+[fmt(next(r for r in P if r['arm']==a and r['split_tag']==t)['R2']) for t in tags] for a in arms],[1.7,1.2,1.2,1.2,1.2],'On Strategy 3, A5 predictions equal the gated-physics predictions because the learned correction is suppressed. Its score is not evidence of a learned gain on that split.')
para(s,'A0 is the no-physics ablation. A1 uses a latent-heat-based physical scale; A2 uses a Katto scale; A3 changes the input representation to dimensionless groups; A5 applies the constrained/gated formulation. This architecture has different correction controls from the later factor-1.5-to-10 flow model. The apparent Strategy 3 improvement from 0.173 in the initial HistGB benchmark to about 0.711 in A1 combines model and data-processing differences and is not attributed solely to unit corrections.')
s.add_heading('S3 Additional flow summaries',1)
L=rows('results/prof_tasks/audit_correction_model.csv')
table(s,'Table S4. Stored residual-learner comparison.',['Learner','Seeds','Median R²','Worst R²','Median within ±10% (%)'],[[r['model'],r['n_seeds'],fmt(r['R2_median']),fmt(r['R2_worst']),fmt(r['w10_median'],1)] for r in sorted(L,key=lambda r:-r['R2_median'])],[1.7,.6,1.1,1.1,2.0],'Learner configurations differ. Deterministic configurations may have one run, and the SVR uses a fitting cap. The table is exploratory and does not establish statistical equivalence or a uniquely optimal learner.')
FL=rows('results/prof_tasks/audit_fluid_transfer.csv')
table(s,'Table S5. Leave-fluid-out summary.',['Held fluid','Test rows','R²','Within ±10% (%)','Low-trust fraction (%)'],[[r['held_fluid'],r['n_test'],fmt(r['PIPE_R2']),fmt(r['PIPE_w10'],1),fmt(r['pct_flagged_low_trust'],1)] for r in FL],[1.1,.9,.9,1.6,2.0],'The trust score contains an explicit unseen-fluid indicator. Low trust is not proof that numerical distance alone recognizes the held fluid.')
data=[]
for did,lab in zip(IDS,LABELS):
 r=getf(did,'70/30');data.append([lab,fmt(r['R2_mean']),fmt(r['R2_sd']),f"{100*r['within_10pct_mean']:.1f}",f"{100*r['coverage90_mean']:.1f}"])
table(s,'Table S6. Ten-seed 70/30 within-dataset evaluation.',['Dataset','Mean R²','Seed SD','Within ±10% (%)','Coverage (%)'],data,[1.7,1.0,1.0,1.4,1.4],'Coverage has a nominal target of 90%. These results are within-source evaluations and do not resolve cross-source transfer failures.')
s.add_heading('S4 Experimental provenance and exclusions',1)
para(s,'The numerical source files bundled with this revision are copies of the stored summaries, not outputs of newly trained models. Their SHA-256 digests identify the exact artifacts used. Relative source paths refer to the research repository. The repository working tree contains changes after commit ec3b2b3, so that commit alone is not a complete identifier of the evaluated research state.')
for f,v in E.items():
 p=para(s,f);p.paragraph_format.keep_with_next=True
 for r in p.runs:setfont(r,10,True)
 p=para(s,'SHA-256: '+v['sha256'])
 for r in p.runs:setfont(r,9)
para(s,'The KAERI uniform XML has MD5 862faef6604bba45a975e4659550a549, matching the retrieved version-2 record. The local nonuniform XML has MD5 bcb16928ef0e105b5e5995cd60213a95, whereas version 2 lists 066ca3ede530e2d4dc293394446e8f8f. No updated dataset was downloaded or substituted during manuscript preparation. The effect of the version difference has not been quantified.')
para(s,'The original lookup-table/neural figures are not retained as comparable final-model evidence. The earlier neural Stage 1 extrapolation result used an 8 MPa cutoff and 13,748 test rows; the later parity plot used 5,994 rows above 14 MPa. Their difference is not seed variation on one test set. The later synthetic-pretraining population also covers pressures above some experimental holdout cutoffs, so experimental extrapolation is not equivalent to absence of all prior information in that pressure range.')
para(s,'Neural LoRA, small pool-boiling adaptation and mixture-of-experts routing are separate exploratory experiments. The misidentified Hardik water population affects earlier neural property features, and those results are not used as valid R-123 evidence. Mean expert-gate probabilities are not classifier accuracy. Existing RF SHAP results are not presented as explanations of the final ExtraTrees model. These exclusions avoid attributing unsupported capabilities to the principal flow pipeline.')
s.add_heading('S5 Figure provenance',1)
para(s,'Figures 1–6 were generated programmatically from the numerical records and the stated implementation. All lettering uses Times New Roman at 10 pt at a width of 165.1 mm (6.5 inches). Separate vector PDF figures embed the font, and the PNG versions are rendered at 1000 dpi. Scaling the figures changes their final lettering size. Figure 2 is a method schematic, not a data-derived performance plot. No experimental observation was synthesized.')
for i in range(1,7):para(s,CAPTIONS[i])
para(s,'The plot data, source-summary copies, checksums and figure-generation script are included in the supporting package. The script only renders archived results; it does not train or alter a model. References cited in this supplement are given in the main manuscript.')
s.save(OUT/'Supplementary_material.docx')

# Make the supporting numerical files self-contained without altering source files.
pd=OUT/'supporting_data';pd.mkdir(exist_ok=True)
for f in E:shutil.copy2(ROOT/f,pd/Path(f).name)
manifest=[['source_path','sha256','bundled_file']]+[[f,v['sha256'],'supporting_data/'+Path(f).name] for f,v in E.items()]
with (OUT/'experiment_manifest.csv').open('w',newline='') as f:csv.writer(f).writerows(manifest)
(OUT/'figure_captions.txt').write_text('\n\n'.join(CAPTIONS.values()))
shutil.copy2(OUT/'_work/make_figures.py',OUT/'figures'/'make_figures.py')
(OUT/'figures'/'evidence.json').write_text(json.dumps(E,indent=2))
(OUT/'AUTHOR_ACTIONS_BEFORE_SUBMISSION.txt').write_text('''AUTHOR ACTIONS BEFORE SUBMISSION

The revised scientific manuscript, highlights and supporting material are complete as a paper-only revision. They do not certify that the underlying scientific validity concerns have been resolved.

AUTHOR INFORMATION STILL REQUIRED
1. Designate the corresponding author from the supplied author list.
2. Supply the CRediT contribution statement. This is intentionally pending your promised input.
3. Confirm the competing-interest declaration. No declaration of no interests has been invented.
4. Confirm permission and terms for sharing the compiled datasets; provide the public code repository URL or DOI when assigned.
5. All authors must review and approve the content, authorship, AI disclosure and submission.

SCIENTIFIC POINTS THAT PAPER EDITING CANNOT SETTLE
- Establish whether nonfinite predictions invoked target-median replacement in the five-seed results, or replace/exclude affected estimates.
- Resolve quality-dependent auxiliary paths and evaluation-versus-deployment equivalence before claiming prospective validation.
- Independently confirm the target-reconstructed helical geometry and review missing-input assumptions.
- Reconcile Katto branch-selection implementation with the original source.
- Reconcile the archived KAERI nonuniform snapshot with the updated source release; the effect on scores is unknown.
- Preserve exact run configurations and row predictions before presenting the study as fully reproducible.

The manuscript reports these limitations explicitly and withdraws unsupported universal, leakage-free, safety and calibration claims. No code changes or training were performed for this revision. A paper-only revision cannot guarantee acceptance or cure invalid experimental evidence.

CONFIRMED DETAILS
Authors, in order: Mauryavardhan Singh; Mayank Nagar; Ayushman Paul; Hardik Mahawar; Bikash Pattanayak.
Affiliation: Indian Institute of Technology Kharagpur, Kharagpur, West Bengal 721302, India.
Funding: no funding, as supplied by the author.
Code: intended for public sharing, as supplied by the author.

TYPOGRAPHY AND FILES
Manuscript body: 12 pt Times New Roman, single column.
Figures: 10 pt Times New Roman at 165.1 mm width; vector PDFs and 1000 dpi PNGs.
Elsevier artwork sizing: https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
Journal guide: https://www.sciencedirect.com/journal/nuclear-engineering-and-design/publish/guide-for-authors
The 10 pt figure choice is a readability choice supported by the general guidance, not a verified recollection of the professor's instruction.
''')
abstract=(OUT/'_work/paper_text.md').read_text().split('# Abstract\n',1)[1].split('\n\nKeywords:',1)[0]
print('Abstract words:',len(abstract.split()));assert len(abstract.split())<=250
print('Highlight lengths:',[len(t) for t in highlights]);print('Created manuscript, highlights and supplement.')
