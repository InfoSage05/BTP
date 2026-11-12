import pandas as pd, numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'data/chf_long_clean.csv')
df=df[df.CHF>0]; df=df[df.X<1.0]
tr=df[df.P<=16000]; te=df[df.P>=17000]
print(len(df),len(tr),len(te))
F=['P','G','X']
et=ExtraTreesRegressor(300,random_state=0,n_jobs=-1).fit(tr[F],tr.CHF)
rf=RandomForestRegressor(300,random_state=0,n_jobs=-1).fit(tr[F],tr.CHF)
rd=make_pipeline(StandardScaler(),PolynomialFeatures(2),Ridge(1.0)).fit(tr[F],np.log(tr.CHF))
for n,p in [('ET',et.predict(te[F])),('RF',rf.predict(te[F])),('Ridge2log',np.exp(rd.predict(te[F])))]:
    print(n,round(r2_score(te.CHF,p),4))
# candidate slices: change of truth from 16MPa to 21MPa
P=sorted(df.P.unique()); print(P)
g=df.pivot_table(index=['G','X'],columns='P',values='CHF')
ch=(g[21000]/g[16000]).sort_values()
print(ch.head(10)); print(ch.tail(5))
for G,X in [(500,-0.2),(500,0.0),(750,-0.2),(1000,0.0),(500,-0.1)]:
    s=df[(df.G==G)&(df.X==X)].sort_values('P')
    print(G,X)
    print(pd.DataFrame({'P':s.P.values,'true':s.CHF.values,'ET':et.predict(s[F]).round(),'Ridge':np.exp(rd.predict(s[F])).round()}).iloc[12:].to_string(index=False))
