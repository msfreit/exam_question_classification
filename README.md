# exam_question_classification
Modelo de Machine Learning para Classificação de Questões do Vestibular


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
	separa em 2 partes
	 - uma parte pra treinar
	 - outra pra testar
	 
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