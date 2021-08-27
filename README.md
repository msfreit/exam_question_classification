# exam_question_classification

Modelo de Machine Learning para Classificação de Questões de Vestibular

# Análise de Questões

## I - Exposição do Problema

Este projeto iniciou-se a partir de um problema da startup em que faço parte. Um certo momento, levantou-se a necessidade de fornecer aos nossos usuários, questões de vestibulares para que eles possam estudar mais e ter mais conteúdos para se darem bem no vestibular.
Assim, criou-se um banco de dados com 120 mil questões
Porém, após validarmos os dados, identificou-se que nem todos as questões estavam classificadas de acordo com o assunto que elas pertenciam.
Assim, surgiu a ideia de implementar um modelo e treiná-lo, a fim de classificar cada questão de acordo com o assunto que essa questão pertence.

### Modelo
Para isso, foi necessário utilizar alguma técnica para a avaliação de textos.
Assim, idealizou-se usar o bag of words. Uma explicação simples é que o bag of words é uma lista que contem todas as palavras que estão nos textos de maneira não repetida.
Utilizamos ela para poder identificar as palavras mais recorrentes e entender se ela agregam na classificação das questões.

## II - Importação dos dados

Como os dados estavam no bando do Mongo. utilizei o framework do MongoDB para conectar e importar os dados para o Python

```
from bs4 import BeautifulSoup

for i in range(len(corpus )):
    corpus[i] = corpus[i].lower()
    corpus[i] = BeautifulSoup(corpus [i]).get_text() # transforma o HTML em texto
    corpus [i] = re.sub(r'\W',' ',corpus [i])  # remove os caracteres especiais
    corpus [i] = re.sub(r'\s+',' ',corpus [i]) # remove os caracteres especiais
    
```

Como parte do processo de processamento de texto, é necessário verificar quais palavras são as mais frequentes, para que sejam usadas posteriormente.
Assim, foi utilizado a função "word_tokenize", que pega uma sentença e separa os dados em posições de uma lista.

```
wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
```

Após criar um array com a frequencia de cada palavra, é necessário eliminar as palavras "irrelevantes" para o nosso problema (stopwords), com o objetivo de reduzir o ruído dos dados analisados. Assim, se a palavra faz parte do conjunto de palavras do meu _stopwords_, altero a frequência da mesma para _zero_.

classificação

agrupar questões

bag of words
n grams
########################################################################

Seguem as ações que conversamos agora:

1. elaborar a árvore com disciplinas e assuntos que serão mapeados na classificação;

2. preparar a base de dados com questões previamente classificadas dentro segundo os assuntos mapeados na árvore;

3. estudar as primeiras técnicas para classificação de texto como bag-of-words e regressão logística;

4. estudar técnicas mais avançadas para classificação de texto.

--- proposições

########################################################################
removo stopwords

procurar biblioteca com stopwords

sigmoide logistica

cross validation:
separa em 2 partes - uma parte pra treinar - outra pra testar
matriz de confusão.
########################################################################

Tópicos da mentoria 05/06

Converter HTML para texto (ex.: UTF-8, ASCII)

Uso da classe CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

Stopwords: https://gist.github.com/alopes/5358189

Validação cruzada:

- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

Matriz de confusão: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

Atividades posteriores:

- validação cruzada (KFold estratificado): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

- bag of words normalizado (TfidfVectorizer): https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

- repositório com códigos de exemplo: https://github.com/alex-carneiro/Moving2DS

############################################################################
26/06/2021
Clustering: https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

Clustering com Scikit-Learn: https://scikit-learn.org/stable/modules/clustering.html

Overfit: https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/

Distância de Levenshtein: http://people.cs.pitt.edu/~kirk/cs1501/Pruhs/Spring2006/assignments/editdistance/Levenshtein%20Distance.htm

Distância de Levenshtein com Python: https://pypi.org/project/python-Levenshtein/

Uso de votação para associar novas tags para as sentenças após clusterização

TODO: elaborar o pipeline sequencial que descreva as tarefas para agrupamento e mapeamento das questões

Vídeo sobre descrição automática de cenas: https://www.youtube.com/watch?v=40riCqvRoMs

############################################################################
10/07/2021
Como definir o número ideal de clusters para o K Means: https://jtemporal.com/kmeans-and-elbow-method/
