bins = [0, 25, 50, 75, 100]
labels = [1,2,3,4]
for i in tabela_dados3.columns[0:4]:
    tabela_dados3[i] = pd.cut(tabela_dados3[i], bins=bins,labels=labels)
tabela_dados3['Silencio']
