
#Importa as bibliotecas necessarias
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.svm import SVC

#Le o data frame
base = pd.read_csv('breast_cancer_classification.csv')

#Delea a coluna com Id
del(base['id'])

#Divide o data frame em atributos e classe
atributos = base.iloc[:, 1:].values
classe = base.iloc[:, 0].values

#Muda as variaveis da classe de nominais para variaveis numericas
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#Aplica o escalonamento
scaler = StandardScaler()
atributos = scaler.fit_transform(atributos)

#Divide os dados para treinamento e teste
atributos_treinamento, atributos_teste, classe_treinamento, classe_teste = train_test_split(atributos, classe, test_size=0.25, random_state=1)

#SVM
classificador = SVC(kernel = 'linear', random_state=1, C=1.0)
classificador.fit(atributos_treinamento, classe_treinamento)
previsoes = classificador.predict(atributos_teste)


#Mostra a precisao de acerto do algoritmo 
#0.9720
precisao = accuracy_score(classe_teste, previsoes)
#Mostra uma matriz de erros e acertos
matriz = confusion_matrix(classe_teste, previsoes)



