import urllib.request
Hydrocarbons=['Ethylene',"Ethane",'Propyne','Propene','Propane','butane',
              '2-methylpropane','Butene','2-methylprop-1-ene','(Z)-but-2-ene','(2E)-6-methylhept-2-ene',
              '3-ethyl-3-methylpent-1-ene','3-methylheptane']
for i in range(len(Hydrocarbons)):
     url1='https://cactus.nci.nih.gov/chemical/structure/{}/smiles'.format(Hydrocarbons[i])
     with urllib.request.urlopen(url1) as url:
        s = url.read()
        print(s)
