import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import pandas as pd
df=pd.read_csv("datacom.csv")
col=["Age","Height","Weight","NCP","bmi","CAEC","FAF"] 
scaler=StandardScaler()
df[col]=scaler.fit_transform(df[col])

x=df.drop(columns="NObeyesdad")
y=df["NObeyesdad"]
sm=SMOTE(sampling_strategy={0:400,1:400},random_state=0)
x, y = sm.fit_resample(x, y)
tm=RandomUnderSampler(sampling_strategy={2:400})
x,y=tm.fit_resample(x,y)
df=pd.concat([x,y],axis=1)

print(df["NObeyesdad"].value_counts())
xfinal=df.drop(columns="NObeyesdad")
yfinal=df["NObeyesdad"]
print(xfinal)
model=XGBClassifier(learning_rate= 0.1, max_depth= 3, n_estimators=150, n_jobs= 3,random_state=0)
model.fit(xfinal,yfinal)

pickle.dump(model,open('model.pkl',"wb"))
pickle.dump(scaler,open('scaler.pkl',"wb"))
