# quick-pandas
Make [pandas](https://pandas.pydata.org/) run faster with a single monkey\_patch call.

# Install
```shell
pip install quick-pandas

```

# Usage
```python
import pandas as pd
from quick_pandas import monkey
monkey.patch_all()

df = pd.DataFrame(data=[1])
df.sort_values(kind='radixsort', by=0)

```

# Notice
This library is still under development and is unstable. Do *NOT* use it unless you know what you are doing. 
