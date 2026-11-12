from pathlib import Path
import json, math
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color, black, white
import pypdfium2 as pdfium

OUT=Path(__file__).resolve().parents[1]
evidence_path=OUT/'figures/evidence.json'
if not evidence_path.exists():evidence_path=OUT/'_work/evidence.json'
E=json.loads(evidence_path.read_text())
(OUT/'_work').mkdir(exist_ok=True)
def rows(name): return E['results/prof_tasks/'+name]['rows']
F=rows('FINAL_results.csv')
IDS=['D1_NRC','D2_Zhao2020','D3_KAERI_uniform','D4_KAERI_nonuniform','D5_Helical_R123','D6_Hardik_helical_water','D7_Hardik_straight_R123','D8_Pioro_R134a']
LABELS=['NRC','Zhao','KAERI uniform','KAERI nonuniform','Helical R-123','Hardik water','Straight R-123','Pioro R-134a']
COUNTS=[24443,1864,651,888,257,156,55,268]
pdfmetrics.registerFont(TTFont('TNR','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
pdfmetrics.registerFont(TTFont('TNRB','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
BLUE=HexColor('#185A82'); ORANGE=HexColor('#B45C16'); GREEN=HexColor('#286B52'); GRAY=HexColor('#666666'); GRID=HexColor('#D8D8D8')
def new(n,h):
 c=canvas.Canvas(str(OUT/'figures'/f'Figure_{n}.pdf'),pagesize=(468,h));c.setTitle(f'Figure {n}');c.setAuthor('Mauryavardhan Singh and coauthors');c.setFont('TNR',10);return c
def text(c,x,y,t,align='left',bold=False,color=black):
 c.setFillColor(color);c.setFont('TNRB' if bold else 'TNR',10)
 getattr(c,{'left':'drawString','right':'drawRightString','center':'drawCentredString'}[align])(x,y,str(t))
def line(c,x1,y1,x2,y2,color=GRID,w=.5,dash=None):
 c.setStrokeColor(color);c.setLineWidth(w);c.setDash(dash or []);c.line(x1,y1,x2,y2);c.setDash([])
def mark(c,x,y,color,shape=0):
 c.setFillColor(color);c.setStrokeColor(color)
 if shape==0:c.circle(x,y,3,fill=1,stroke=0)
 elif shape==1:c.rect(x-3,y-3,6,6,fill=1,stroke=0)
 else:
  p=c.beginPath();p.moveTo(x,y+3.8);p.lineTo(x-3.6,y-2.8);p.lineTo(x+3.6,y-2.8);p.close();c.drawPath(p,fill=1,stroke=0)
def legend(c,items,y=12):
 x=12
 for label,col,sh in items:
  mark(c,x+3,y+3,col,sh);text(c,x+11,y,label);x+=pdfmetrics.stringWidth(label,'TNR',10)+31
def finish(c,n):
 c.showPage();c.save();pdf=pdfium.PdfDocument(str(OUT/'figures'/f'Figure_{n}.pdf'))
 page=pdf[0]; im=page.render(scale=1000/72).to_pil();im.save(OUT/'figures'/f'Figure_{n}.png',dpi=(1000,1000));
 preview=page.render(scale=2.4).to_pil();preview.save(OUT/'_work'/f'Figure_{n}_preview.png');page.close();pdf.close()

# Figure 1: a logarithmic count axis is explicit; no zero-count implication.
c=new(1,270);x0=148;w=276;top=234;step=23
for v in [1,10,100,1000,10000]:
 x=x0+w*math.log10(v)/4.5;line(c,x,48,x,245);text(c,x,34,f'{v:,}','center')
for i,(lab,n) in enumerate(zip(LABELS,COUNTS)):
 y=top-i*step;col=BLUE if i in [0,1,2,3,5] else ORANGE
 text(c,x0-9,y-3,lab,'right');c.setFillColor(col);c.rect(x0,y-5,w*math.log10(n)/4.5,10,stroke=0,fill=1)
 text(c,x0+w*math.log10(n)/4.5+5,y-3,f'{n:,}')
text(c,x0+w/2,19,'Number of records (logarithmic scale)','center')
legend(c,[('Water',BLUE,0),('Refrigerants',ORANGE,0)],4);finish(c,1)

# Figure 2: scientific method schematic, with retrospective information visible.
c=new(2,302)
def box(x,y,w,h,lines,color=BLUE):
 c.setFillColor(white);c.setStrokeColor(color);c.setLineWidth(.8);c.roundRect(x,y,w,h,4,stroke=1,fill=1)
 cy=y+h/2+(len(lines)-1)*6
 for j,s in enumerate(lines):text(c,x+w/2,cy-j*12-3,s,'center',j==0)
def arrow(x1,y1,x2,y2,color=GRAY):
 line(c,x1,y1,x2,y2,color,.9);ang=math.atan2(y2-y1,x2-x1)
 for d in [-.55,.55]:line(c,x2,y2,x2-6*math.cos(ang+d),y2-6*math.sin(ang+d),color,.9)
ys=[250,195,140,85,30]
labs=[['Source data and properties','Units, geometry and inlet assumptions'],['Five dimensionless inputs','Physics predictions and source labels'],['Training partition','Selected backbone plus ExtraTrees residual'],['Trust dependent correction bound','Factor 1.5 at zero trust to 10 at full trust'],['Retrospective prediction and interval','Point errors and empirical coverage']]
for y,ls in zip(ys,labs):box(9,y,260,42,ls)
for a,b in zip(ys,ys[1:]):arrow(139,a,139,b+42)
box(291,210,168,72,['Local quality in archived data','Used by LUT and Biasi arms','and correlation disagreement'],ORANGE)
arrow(291,226,269,216,ORANGE)
box(291,119,168,66,['Separate calibration slice','20% of outer training rows','Minimum 30 rows'],GREEN)
arrow(269,161,291,151,GREEN);arrow(375,119,375,51,GREEN);arrow(375,51,269,51,GREEN)
text(c,234,10,'Evaluated pathways do not establish an inlet only deployment guarantee','center');finish(c,2)

# Figure 3: comparable accuracy unit across protocols, no R2 scale truncation.
c=new(3,293);x0=143;w=304;top=252;step=25
for v in [0,20,40,60,80,100]:
 x=x0+w*v/100;line(c,x,49,x,268);text(c,x,35,str(v),'center')
for i,(d,lab) in enumerate(zip(IDS,LABELS)):
 y=top-i*step;text(c,x0-9,y-3,lab,'right')
 for kind,off,col,sh in [('80/20',5,BLUE,0),('GROUPED',0,ORANGE,1),('LODO',-5,GREEN,2)]:
  found=[r for r in F if r['dataset']==d and r['split'].startswith(kind)]
  if found:mark(c,x0+w*found[0]['within_10pct_mean'],y+off,col,sh)
text(c,x0+w/2,20,'Predictions within ±10% of measured CHF (%)','center')
legend(c,[('Random 80/20',BLUE,0),('Grouped',ORANGE,1),('Leave one dataset out',GREEN,2)],4);finish(c,3)

# Figure 4: final five-seed estimates; bars are seed SD, not confidence intervals.
c=new(4,290);top=244;step=25;x0=145;w=176;x1=355;w1=95
text(c,x0+w/2,274,'(a) R² with seed standard deviation','center',True)
text(c,x1+w1/2,259,'(b) Within ±10%','center',True)
for v in [-.5,0,.5,1]:
 x=x0+(v+.6)/1.65*w;line(c,x,49,x,257,GRAY if v==0 else GRID,.7 if v==0 else .5);text(c,x,34,str(v),'center')
for v in [0,25,50,75]:
 x=x1+w1*v/75;line(c,x,49,x,251);text(c,x,34,str(v),'center')
for i,(d,lab) in enumerate(zip(IDS,LABELS)):
 r=next(r for r in F if r['dataset']==d and r['split'].startswith('LODO'));y=top-i*step
 text(c,136,y-3,lab,'right');xx=lambda v:x0+(v+.6)/1.65*w
 a,b=xx(r['R2_mean']-r['R2_sd']),xx(r['R2_mean']+r['R2_sd'])
 line(c,a,y,b,y,BLUE,1);line(c,a,y-3,a,y+3,BLUE,1);line(c,b,y-3,b,y+3,BLUE,1);mark(c,xx(r['R2_mean']),y,BLUE)
 mark(c,x1+w1*r['within_10pct_mean']*100/75,y,GREEN)
text(c,x0+w/2,18,'Coefficient of determination','center');text(c,x1+w1/2,18,'Accuracy (%)','center');finish(c,4)

# Figure 5: matched anchored ablation; direct ML values retained in its table.
c=new(5,293);x0=143;w=304;top=252;step=25
for v in [-.75,-.5,-.25,0,.25,.5,.75,1.0]:
 x=x0+w*(v+.8)/1.85;line(c,x,49,x,268,GRAY if v==0 else GRID);text(c,x,35,f'{v:g}','center')
A=[r for r in rows('audit_v2_features.csv') if r['set']=='F0_base5']
for i,(d,lab) in enumerate(zip(IDS,LABELS)):
 y=top-i*step;text(c,x0-9,y-3,lab,'right')
 for arm,off,col,sh in [('backbone_alone',5,GRAY,0),('PIPE_unbounded',0,ORANGE,1),('PIPE',-5,BLUE,2)]:
  r=next(r for r in A if r['held_out']==d and r['arm']==arm);mark(c,x0+w*(r['R2']+.8)/1.85,y+off,col,sh)
text(c,x0+w/2,20,'Leave one dataset out R² (three seed means)','center')
legend(c,[('Backbone',GRAY,0),('Unbounded residual',ORANGE,1),('Bounded residual',BLUE,2)],4);finish(c,5)

# Figure 6: same single-seed audit for all three interval constructions.
c=new(6,310);x0=143;w=304;top=268;step=25
for v in [0,20,40,60,80,100]:
 x=x0+w*v/100;line(c,x,62,x,284);text(c,x,48,str(v),'center')
line(c,x0+.9*w,62,x0+.9*w,284,GRAY,.9,[3,3]);text(c,x0+.9*w-3,289,'90% nominal','right')
S=rows('audit_ood_conformal_strict.csv')
for i,(d,lab) in enumerate(zip(IDS,LABELS)):
 y=top-i*step;text(c,x0-9,y-3,lab,'right');r=next(r for r in S if r['held_out']==d)
 for key,off,col,sh in [('cov_shipped',5,GRAY,0),('cov_strict',0,ORANGE,1),('cov_strict_trust',-5,BLUE,2)]:mark(c,x0+w*r[key],y+off,col,sh)
text(c,x0+w/2,32,'Empirical prediction interval coverage (%)','center')
legend(c,[('Default',GRAY,0),('Pairwise source exclusion',ORANGE,1)],18)
legend(c,[('Pairwise source exclusion plus widening',BLUE,2)],4);finish(c,6)
print('Created 6 PDF figures with embedded Times New Roman and 1000 dpi PNG counterparts.')
