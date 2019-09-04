for f in features:
    df1 = df[[f]]

    df2 = (pd.get_dummies(df1, prefix='', prefix_sep='')
                   .max(level=0, axis=1)
                   .add_prefix(f+' - '))  
    dataframe = pd.concat([df, df2], axis=1)
    
    dataframe.select_dtypes(include=['floating','integer'])
    
dataframe3.groupby(['NomeFantasia','Rede','Categoria',pd.Grouper(key='DateConvert', freq='W-SUN')]).sum().sort_values(['NomeFantasia','DateConvert'])

df6.groupby(['NomeFantasia','Rede','Categoria'], as_index=False).mean().sort_values(['NomeFantasia'])

cor = x.corr()
cor_target = abs(cor["Revenue"])
selected=cor_target[(cor_target>0.5) & (cor_target<0.9)].reset_index().iloc[:,0]
selected

semana_que_vem=int(pd.Series(data_inicio_previsao).dt.strftime('%U'))+1

data_inicio_previsao + pd.DateOffset(1)
