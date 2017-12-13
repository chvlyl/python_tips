# Python Tips
A collection of python tips in my everyday coding

## Python

## Numpy

## Pandas

#### 1. Pandas groupby excludes NA

```python
df = pd.DataFrame({'col1':[1,2,np.NaN,1,np.NaN,2],'col2':[0,0,1,2,2,2]})
df.GroupBy(['col1']).size()
```

It will exclude the NA in col1. We can fill the Nan with string "NA".

```python
df = pd.DataFrame({'col1':[1,2,np.NaN,1,np.NaN,2],'col2':[0,0,1,2,2,2]})
df.fillna('NA').GroupBy(['col1']).size()
```
#### 2. Pandas rename columns

We can rename the entire column names by 
```python
df.columns = ['a', 'b']
```

Or we can rename specific columns by
```python
df.rename(columns = {'a':'b'}, inplace = True)
```

## Keras

## Other

### 1. Use R in Jupyter Notebook
Install some R packages in R console first.
```R
install.packages(c('repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))
devtools::install_github('IRkernel/IRkernel')
IRkernel::installspec()
### if you want to install other R packages 
### make sure to add the new package to the anaconda R library path
##install.packages("newpackage", "/home/user/anaconda3/lib/R/library")
```

Use conda to install some necessary R packages in your Python env.
```
### if you want to run R in Jupyter Notebook
conda install -c r r-essentials
### if you want to run both R and Python in the same notebook
conda install -c r rpy2
```

Run both R and Python code in the same notebook. Note that the Python code and R code need to be in separated cells.
```python
%load_ext rpy2.ipython
```
```python
%R require(ggplot2)
```
```python
import pandas as pd 
df = pd.DataFrame({'group': ['a', 'b', 'c', 'd','e', 'f', 'g', 'h','i'],
                   'A': [1, 2, 5, 5, 1, 6, 7, 5, 9],
                   'B': [0, 2, 3, 6, 7, 6, 5, 9, 13],
                   'C': [3, 2, 3, 1, 3, 3, 4, 2, 1]})
```
```python
%%R -i df
ggplot(data=df) + geom_point(aes(x=A, y=B, color=C))
```
