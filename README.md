# Python Tips
A collection of python tips in my everyday coding



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
