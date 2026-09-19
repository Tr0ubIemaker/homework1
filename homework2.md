![IMG_20260919_131713.](C:\Users\Administrator\Downloads/IMG_20260919_131713..jpg)





(R^2\)为什么上升 / 不变 \(R^2=1-\frac{RSS}{TSS}\)，TSS 固定，新增变量后\(RSS\downarrow\)，所以\(R^2\)上升；若该随机变量完全和 y 正交，RSS 不变，则\(R^2\)不变。n样本量；k为解释变量个数；残差自由度 \(df_{resid}=n-k-1\)加入无关随机特征，k增大，残差自由度\(n-k-1\)变小；虽然 RSS 略微下降，但分母\(RSS/(n-k-1)\)（残差均方）会变大。(TSS/(n-1)\)不变，所以 \(1-\frac{RSS/(n-k-1)}{TSS/(n-1)}\) 变小，即\(R^2_{adj}\)下降。





```python
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

Carseats csv
url = "https://raw.githubusercontent.com/selva86/datasets/master/Carseats.csv"
carseats = pd.read_csv(url)

model = smf.ols(formula='Sales ~ Price + Income + Advertising + ShelveLoc', data=carseats).fit()

print(model.summary())

print("\n基准组 ShelveLoc：Bad")
print(f"ShelveLoc[T.Good] 系数 = {model.params['ShelveLoc[T.Good]']:.4f}")

exog = model.model.exog
names = model.model.exog_names
vif_list = []
for idx, n in enumerate(names):
    if n != "Intercept":
        vif_list.append({"var":n, "VIF":variance_inflation_factor(exog, idx)})

print(pd.DataFrame(vif_list).round(3))
```

