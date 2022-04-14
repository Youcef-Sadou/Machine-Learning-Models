# Machine-Learning-Models
Ce readme est disponible en deux langues: [Anglais](#english) et [Français](#français)
This readme has two languages : [English](#english) and [French](#français)


## Français:

### Implémentation d'un arbre de régression

Le premier ensemble de données est un ensemble de ventes de jeux vidéo et il a ces attributs

Nous importons d'abord le fichier csv dans un data frame à l'aide de la bibliothèque pandas

![](/screenshots/s2.png)

#### 1.1 Prétraitement des données

Maintenant nous commençons le prétraitement des données, d'abord nous identifions l'attribut que nous
voulons prédire, nous avons choisi les ventes de jeux vidéo européens, maintenant nous identifions les
attributs "inutiles" à notre module d’apprentissage, qui sont le nom, le rang et le global_sales (nous avons
supprimé le dernier car si nous le laissons le module peut simplement tout ignorer et faire une simple
soustraction à trouver les eu_sales)

![](/screenshots/s3.png)

Maintenant que nous nous sommes débarrassés des attributs indésirables, nous devons examiner de plus
près les données et trouver des éléments de données manquants les attributs.

![](/screenshots/s4.png)

Ici on peut voir qu'il manque une année de jeux vidéo et un éditeur. Nous avons deux manières de traiter ce
problème de manquant :
1) on remplace l'attribut manquant par la moyenne de toutes les données d'un même attribut (moyenne de
l'année) et pour l'éditeur manquant on peut le remplacer par l'éditeur le plus courant (en termes
d'occurrence), ainsi le module va ne pas être trop affecté lors de l'utilisation de la moyenne
2) la deuxième méthode consiste à supprimer les données avec des attributs manquants, nous utiliserons
cette méthode car les données manquantes sont si petites qu'elles sont fondamentalement négligeables (elles
représentent 0,01 pour cent de l'ensemble de données total)

![](/screenshots/s5.png)

Maintenant que nous avons pris les données manquantes, l'étape suivante du prétraitement des données est
la transformation des attributs d'objet ou ce qu'on appelle la création de variables factices, nous pouvons
prendre un exemple d'attribut tel que Online_availablity qui prend soit "YES" soit "NO" pour la valeur,
nous pouvons le transformer en Online_availablity_YES qui prend 1 s'il est disponible en ligne et 0 s'il ne
l'est pas.
En général pour les attributs d'objet avec N valeurs différentes, nous les transformons en N-1 variables
fictives, dans notre exemple nous avons 3 "plateforme", "genre" et "Publisher"

![](/screenshots/s6.png)

Maintenant avec cela fait, nous avons plus de 600 attributs.
28l'étape finale consiste à diviser le cadre de données en deux parties, nos variables x qui sont nos attributs
indépendants (année, éditeur, ...) et la variable y qui est notre variable cible ou variable dépendante (ventes
UE) et nous pouvons commencer l'entrainement.

![](/screenshots/s7.png)

#### 1.2 L'entrainement de model

Nous sommes intéressés par l'exactitude de ces prédictions lorsque nous appliquons notre méthode à des
données de test inédites. Ce que nous allons faire, c'est diviser nos données en deux parties, l'une sera
appelée ensemble d'apprentissage, cela sera utilisé pour entraîner le modèle, et l'autre partie sera l'ensemble
de test, ce seront les données invisibles et elles seront utilisées pour évaluer la précision de notre modèle
, les données qui ne sont pas utilisées pour entraîner notre modèle donc en pratique, nous utilisons
généralement 80% de nos données disponibles et nous prenons 20% de nos données disponibles comme
données de test, nous ne testons notre modèle que sur ces données et comparons les performances de nos
différents modèles sur ces données de test pour évaluer nos modèles sur des données réelles, nous avons
donc 16291 lignes de données, nous conserverons 20% de ces données comme données de test et nous
entraînerons notre modèle sur les 80 pour cent de ces données, cette séparation de nos données en test et
train est connue sous le nom de fractionnement du train de test et c'est très facile.

![](/screenshots/s8.png)

pour effectuer le test de fractionnement du train à l'aide de sklearn, nous importons d'abord test_train_split
à partir de model_selection, maintenant cette méthode test_train_split prendre la valeur X, la valeur Y, la
taille de nos données de test (nous avons dit que nous prenons 20% comme données de test, nous avons
donc fourni 0,2) puis il y a un autre paramètre qui est random_state puisque nous attribuons au hasard nos
données dans le test et l'entraînement, pour obtenir les mêmes données de test à chaque fois afin que nous
puissions comparer les performances de notre modèle nous pouvons utiliser cette variable random_state,
c'est juste un nombre aléatoire, nous pouvons utiliser n'importe quelle valeur que nous voulons, l'avantage
de ce random_state est que si nous gardons cette valeur la même tout au long du Programme, nous
obtiendrons exactement la même répartition à chaque fois. Vous pouvez voir que cet échantillon ressemble
exactement à l'échantillon de la trame de données X, nous n'avons aucune variable Y ici, nous n'avons que
X et une chose à noter ici sont les index, vous pouvez voir que nos index sont mélangés maintenant puisque
certains lignes vont dans train_data et certaines des lignes vont dans test_data, nous pouvons voir que les
données d'apprentissage sont effectivement à 80% (13032)

![](/screenshots/s9.png)

Nous devons d'abord importer tree depuis sklearn puis nous créons un objet regtree ici regtree est notre
nom de variable et nous utiliserons tree.DecisionTreeRegeressor() il y a plusieurs paramètres à l'intérieur,
et nous entraînons le regtree en utilisant la méthode fit.

![](/screenshots/s10_1.png)

![](/screenshots/s10_2.png)

maintenant que nous avons entraîné notre modèle à l'aide des données d'entraînement, nous pouvons
prédire des valeurs à l'aide des données de test maintenant que nous avons créé ce y_train_pred et
y_test_pred nous allons utiliser ces valeurs prédites pour calculer la performance de notre modèle, nous ne
pouvons utiliser que mse (erreur quadratique moyenne) pour calculer les performances de différents
modèles sur le même ensemble de données, pour notre ensemble d'entraînement, nous avons 0.74 donc
notre r2_score est de 0,74, ce qui signifie que notre modèle fonctionne très bien, calculons les valeurs r2 sur
nos données de test, alors n'oubliez pas que nous avons entraîné notre modèle sur ces données
d'entraînement et que nous ne pouvons pas l'utiliser les données d'entraînement pour évaluer les
performances que nous devons fournir aux données de test que nous avons conservées lors de
l'entraînement de notre modèle, nous devrions toujours regarder les valeurs de test r2 pour évaluer les
performances de votre modèle car nous n'avons pas utilisé ces données de test pendant l’ entraînement de
modèle et nous obtenons un meilleur score sur notre ensemble de données d'entraînement (0,74) par
opposition à notre ensemble de test (0,70) car il est évident que nous avons entraîné notre modèle sur notre
ensemble d'entraînement et il sera toujours plus performant sur l'ensemble de données sur lequel il s'est
entraîné

![](/screenshots/s11.png)

#### 1.3 Afficher l'arbre 
La première étape consiste à créer un fichier de points, puis nous devons convertir ce fichier de points en
une image, puis nous utiliserons cette image pour créer un graphique comme on peut le voir , le graphe est
un arbre avec chaque nœud ayant : 1- le X[i] est la variable ieme de l'arbre et x[i]<554 par exemple est la
condition si la condition est vérifiée, nous irons au nœud de gauche, sinon nous irons au nœud de droite,
nous aurons ensuite le nombre d'échantillons dans le nœud. notez que la somme de toutes les valeurs
d'échantillon des nœuds dans un certain niveau sera égale à la valeur d'échantillon dans le nœud racine, et
nous avons la valeur moyenne des nœuds mse est l'erreur quadratique moyenne

![](/screenshots/s12.png)

#### 1.4 Élagage des arbres

Maintenant les arbres de décision qui sont trop gros ont 2 problèmes, ils sont difficiles à interpréter et ils
surdimensionnent les données d'apprentissage donnant ainsi mauvaise performance de test, là pour nous
avons décidé de contrôler la croissance des arbres, l'élagage des arbres, dans cette stratégie nous dessinons
un très grand arbre puis taillez-le, ou vous pouvez dire que nous en coupons des sous-parties qui ne sont
pas bénéfiques afin d'obtenir un sous-arbre optimal.
il y a 3 façons
1) la première consiste à diminuer le nombre maximum de niveaux autorisés dans notre arbre

![](/screenshots/s13.png)

2) la seconde est que nous augmentons le nombre minimum d'échantillons aux nœuds internes (non
feuilles)

![](/screenshots/s14.png)

3) la troisième façon consiste à augmenter le nombre minimum d'échantillons aux nœuds feuilles

![](/screenshots/s15.png)

Nous pouvons également combiner les trois conditions pour effectuer un élagage plus précis.


### 2 Arbre de classification 

Jusqu’à présent, nous avons discuté des arbres de régression, c'est-à-dire que nous essayons de prédire une
variable quantitative continue comme lanombre de ventes de jeux vidéo, nous allons maintenant discuter
des arbres de classification dans lesquels tentera de prédire la variable catégorielle comme, personne ait eu
ou non une crise cardiaque.
Première différence entre les arbres de régression et les arbres de classification, on trouve la moyenne de la
variable pour obtenir levariable prédite, mais pour les arbres de classification, nous utiliserons le mode,
c'est-à-dire que nous attribuons cette classe à la région qui est la plus classe courante dans cette région. les
arbres de regressoin et la classification utilisent le fractionnement binaire récursif.

#### 2.1 Implémentation d'un arbre de régression 
Pour l'arbre de classification, nous avons choisi un ensemble de données qui contient différentes données
de crise cardiaque (âge, sexe de la personne, déjà marié, maladie cardiaque, taux de graisse, statut
tabagique... etc.) et il prédit s'ils ont eu ou non un accident vasculaire cérébral avant.
34Nous ne reviendrons pas sur le prétraitement cette fois pour des raisons de longueur (tout sera dans les
scripts fournis).
##### Prédiction et performance

![](/screenshots/s16.png)

dans l'arbre de régression, nous avons utilisé r2 et mse pour calculer les performances de notre modèle,
pour les modèles de classification, nous pouvons utiliser precision_score et la matrice de confusion
precision_score est le pourcentage d'enregistrements que nous pouvons identifier correctement à l'aide de
notre modèle chaque fois que nous exécutons la matrice de confusion, nous obtenons une matrice 2*2la
première cellule [0]0] représente le vrai négatif, ce qui signifie que la valeur prédite de l'absence d'AVC
(0) a été correctement prédite la deuxième cellule [1]0] représente un faux négatif, ce qui signifie que la
valeur prédite de l'absence d'AVC (0) a été mal prédite la troisième cellule [1]0] représente un faux positif,
ce qui signifie que la valeur prédite de l'AVC (1) a été mal préditela troisième cellule [1]0] représente un
vrai positif, ce qui signifie que la valeur prédite de l'AVC (1) a été correctement préditenotez que nous
pouvons calculer le precision_score à partir de la matrice, nous divisons le (vrai négatif + vrai positif) par la
somme de matrice par exemple ici nous avons (479+0) / (479+2+30+0) ce qui nous donne un score de 0.93.

#### 2.2 Afficher l'arbre 

![](/screenshots/s17.png)

Ici chaque cellule contient quatre valeurs contient d'abord la condition ensuite nous avons la valeur gini
pour ce seau, ensuite nous avons la taille de l'échantillon de chaque nœud.
Notez que la somme de toutes les valeurs d'échantillon des nœuds dans un certain niveau sera égale à la
valeur d'échantillon dans le nœud racine et la valeur montre les valeurs du bucket, nous avons donc 4380 et
219 1 dans le nœud racine par exemple.
Quant à la couleur l'orange représente une grande pureté de 0 et le bleu représente une grande pureté de 1


#### 2.3 Conclusion sur les arbres

Les arbres de décision pour la régression et la classification présentent un certain nombre d'avantages par
rapport aux approches classiques telles que régression comme :
1) les arbres sont très faciles à expliquer aux gens
2) les arbres reflètent la prise de décision humaine (rationalité) plus étroitement que les autres approches de
régression et de classification
3) ils peuvent être affichés graphiquement, et sont facilement interprétés même par des non-experts
Pour les inconvénients, il y a 1 inconvénient majeur, un arbre de décision simple n'a généralement pas la
même précision prédictive que certaines des autres approches de régression et de classification mais en
agrégeant (combinant) de nombreux arbres de décision ou ce que nous appelons en créant des ensembles,
nous pouvons améliorer considérablement les performances des arbres de décision.

#### 2.4 Apprentissage d'ensemble 


Le problème avec les arbres de décision est que les arbres de décision ont une variance élevée, ce qui
signifie que si j'ai un ensemble de données et que je divise l'ensemble de données en deux parties et j'utilise
chaque partie pour entraîner un modèle, les deux modèles seront très différents en général.
Donc un moyen naturel de réduire la variance et donc d'augmenter la précision de la prédiction est de
prendre de nombreux ensembles d'apprentissage de la population puis construisez un modèle séparé en
utilisant chaque ensemble d'entraînement, enfin nous faisons la moyenne de la prédiction résultante pour
obtenir la prédiction finale.


#### 2.4.1 Bootsrapping

Laissez-nous comprendre comment le bootsrapping nous aide à créer différents échantillons en utilisant le
même ensemble d'échantillons supposons que j'ai un ensemble de données de 5 nombres
7 9 5 4 3 Je souhaite créer 3 sets d'entraînement à partir de cet ensemble d'entraînement. la méthode
consiste à choisir au hasard un nombre dans cet ensemble d'entraînement et à l'ajouter à mon ensemble
d'entraînement afin que nous ayons
9 5 4 3 4
7 9 5 4 7
8 9 9 4 3

#### 2.4.2 Ensachage

1. Pendant l'ensachage, la taille n'est pas effectuée, des arbres de pleine longueur sont cultivés.
2. les arbres individuels ont une variance élevée et un faible biais, la moyenne réduit la variance.
3. en régression, on prend la moyenne des valeurs prédites.
374. dans la classification, nous prenons le vote majoritaire, c'est-à-dire que la valeur la plus prédite sera
considérée comme la prédiction finale.

#### 2.4.3 Boosting

Maintenant nous allons discuter de Boosting, en Boosting nous créons un certain nombre d'arbres mais la
différence est que les arbres sont cultivés de manière séquentielle, cela signifie que chaque arbre est cultivé
à l'aide d'informations provenant d'arbres déjà cultivés, nous discuterons de 2 techniques de renforcement,
le boosting du gradient et le boosting Ada

#### 2.4.4 Boosting du gradient

Boosting du gradient est une procédure d'apprentissage lente qui consiste à adapter notre arbre en utilisant
les résidus actuels plutôt que le résultat obtenu réponse d'abord nous formons l'arbre puis nous trouvons la
différence entre prédit et réel, ceux-ci sont appelés résidus Ensuite, nous utilisons ces résidus pour ajuster
un petit arbre, mais notons que nous contrôlons la longueur en boostant contrairement à l'ensachage où
nous créons des arbres de pleine longueur. Donc ce petit arbre, qui est ajusté sur les résidus est appliqué
avec un paramètre de retrait puis est ajouté à l'arbre d'origine donc le deuxième arbre est essentiellement la
somme du premier arbre et de l'arbre nouvellement créé multipliée par un paramètre de rétrécissement
maintenant en utilisant le deuxième arbre, nous trouvons à nouveau les résidus et en utilisant à nouveau les
résidus, nous ajustons à nouveau un petit arbre sur ces résidus ce petit arbre est à nouveau multiplié en
lambda et ainsi de suite, donc au lieu d'apprendre en créant l'arbre entier en une seule fois, l'amplification
du gradient apprend lentement en créant un petit arbre à la fois, c'est pourquoi on l'appelle un apprenant lent

![](/screenshots/s18.png)

#### 2.4.5 Boosting Ada 
ada boosting ou adaptive boosting dans ce premier, nous créons un arbre, nous découvrons les prédictions
en utilisant cet arbre, où que ce soit cet arbre a été mal classé ou partout où le résidu de cet arbre est très
grand, nous augmentons l'importance de cette observation particulière puis nous créons à nouveau un arbre
sur nos observations maintenant cette fois puisque nous avons augmenté l'importance de ces observations
notre arbre essaiera de capturer ou de classer correctement ces observations afin que cette fois nous
obtenions un deuxième arbre qui sera un peu différent du premier arbre encore une fois, nous trouverons les
résidus ou les observations mal classées, nous augmenterons le poids de ces observations et exécuterons le
modèle encore une fois, de cette façon, nous continuerons à construire le modèle pendant un certain nombre
de temps prédéterminé.

![](/screenshots/s19.png)

### 2.5 Bayes naïfs
Voici quelques notations
P(A) signifie la probabilité de A, P (A/B) signifie la probabilité de A étant donné que B s'est déjà produit
La théorie de bayes dit que P (A/B) = P (B/A) * P(A) / P(B)
Revenant maintenant à notre exemple de crise cardiaque, nous utiliserons le modèle naïf de Bayes pour
prédire la probabilité qu'une personne fasse une crise cardiaque compte tenu de tous les attributs qui
ressemblent à P (crise cardiaque / genre & hypertension & âge & niveau de glucose & .. ..)
Maintenant la raison pour laquelle nous l'appelons naïf est que nous faisons l'hypothèse naïve que les
variables sont indépendantes.
Deux événements A et B sont indépendants si P (A/B) = P(A) et P (B/A) = P(B)

##### Bayes naïfs
![](/screenshots/s20.png)

## Conclusion

Pour réussir à créer un bon modèle d’apprentissage automatique il a nécessité deux choses
fondamentales : la première est de bien comprendre comment l’algorithme choisi fonctionne
mais il n’est pas nécessaire de rentrer dans les détails mathématiques poussées, la deuxième
chose est d’avoir une bonne connaissance sur les prétraitements des données.

## English:

### Implementing a regression tree

The first dataset is a set of video game sales and it has these attributes

First we import the csv file into a data frame using the pandas library

![](/screenshots/s2.png)

#### 1.1 Data preprocessing

Now we start data preprocessing, first we identify the attribute we
want to predict, we chose European video game sales, now we identify the
"useless" attributes to our learning module, which are name, rank and global_sales (we have
deleted the last one because if we leave it the module can just ignore everything and do a simple
subtraction to find the eu_sales)

![](/screenshots/s3.png)

Now that we've gotten rid of the unwanted attributes, we need to take a closer look
close the data and find missing data elements attributes.

![](/screenshots/s4.png)

Here we can see that a year of video games and a publisher are missing. We have two ways of dealing with this
missing problem:
1) the missing attribute is replaced by the average of all the data of the same attribute (average of
the year) and for the missing publisher it can be replaced by the most current publisher (in terms
of occurrence), so the modulus will not be affected too much when using the average
2) the second method is to delete the data with missing attributes, we will use
this method because the missing data is so small that it is basically negligible (they
represent 0.01 percent of the total data set)

![](/screenshots/s5.png)

Now that we have taken the missing data, the next step in data preprocessing is
transforming object attributes or so called creating dummy variables, we can
take an example attribute such as Online_availablity which takes either "YES" or "NO" for the value,
we can turn it into Online_availablity_YES which takes 1 if available online and 0 if not
is not.
In general for object attributes with N different values, we transform them into N-1 variables
fictitious, in our example we have 3 "platform", "genre" and "Publisher"

![](/screenshots/s6.png)

Now with that done, we have over 600 attributes.
the final step is to split the data frame into two parts, our x variables which are our attributes
independents (year, publisher, ...) and the variable y which is our target variable or dependent variable (sales
UE) and we can start training.

![](/screenshots/s7.png)

#### 1.2 Model training

We are interested in the accuracy of these predictions when we apply our method to
unpublished test data. What we are going to do is split our data into two parts, one will be
called the training set, this will be used to train the model, and the other part will be the set
of test, it will be the invisible data and it will be used to evaluate the accuracy of our model, 
data that is not used to train our model so in practice we use
usually 80% of our available data and we take 20% of our available data as
test data, we only test our model on this data and compare the performance of our
different models on this test data to evaluate our models on real data, we have
so 16291 rows of data, we will keep 20% of this data as test data and we
will train our model on the 80 percent of this data, this separation of our data in test and
train is known as test train splitting and it is very easy.

![](/screenshots/s8.png)

to perform the train split test using sklearn, we first import test_train_split
from model_selection, now this method test_train_split take value X, value Y,
size of our test data (we said we take 20% as test data, we have
so provided 0,2) then there is another parameter which is random_state since we are randomly assigning our
data in test and training, to get the same test data every time so that we
can compare the performance of our model we can use this variable random_state,
it's just a random number, we can use any value we want, the upside
of this random_state is that if we keep this value the same throughout the Program, we
will get exactly the same distribution each time. You can see that this sample looks like
exactly to sample data frame X, we don't have any Y variable here, we only have
X and one thing to note here are indexes, you can see our indexes are mixed up now since
some of the rows go into train_data and some of the rows go into test_data, we can see that the
training data is actually at 80% (13032)

![](/screenshots/s9.png)

First we need to import tree from sklearn then we create a regtree object here regtree is our
variable name and we will use tree.DecisionTreeRegeressor() there are several parameters inside,
and we train the regtree using the fit method.

![](/screenshots/s10_1.png)

![](/screenshots/s10_2.png)

Now that we've trained our model using the training data, we can
predict values using test data now that we have created this y_train_pred and
y_test_pred we will use these predicted values to calculate the performance of our model, we don't
can use that mse (root mean square error) to calculate the performance of different
models on the same dataset, for our training set we have 0.74 so
our r2_score is 0.74, which means our model works very well, let's calculate the r2 values on
our test data, so remember that we trained our model on this data
training and that we cannot use the training data to evaluate the
performance that we have to provide to the test data that we have kept during
training our model, we should always look at the r2 test values to assess the
performance of your model because we did not use this test data during training of
model and we get a better score on our training dataset (0.74) by
opposition to our test set (0.70) because it is obvious that we trained our model on our
training set and it will always perform better on the dataset it trained on
trained

![](/screenshots/s11.png)

#### 1.3 Show tree
The first step is to create a points file, then we need to convert this points file to
an image, then we will use this image to create a graph as we can see, the graph is
a tree with each node having: 1- the X[i] is the ith variable of the tree and x[i]<554 for example is the
condition if the condition is verified, we will go to the left node, otherwise we will go to the right node,
then we will have the number of samples in the node. note that the sum of all the values
sample value of the nodes in a certain level will be equal to the sample value in the root node, and
we have the mean value of mse nodes is the mean squared error

![](/screenshots/s12.png)

#### 1.4 Tree pruning

Now decision trees that are too big have 2 problems, they are hard to interpret and they
oversize the training data thus giving poor test performance, there for us
decided to control the growth of trees, tree pruning, in this strategy we draw
a very large tree and then prune it, or you can say we cut off sub-parts of it that are not
not beneficial in order to obtain an optimal subtree.
there are 3 ways
1) the first is to decrease the maximum number of levels allowed in our tree

![](/screenshots/s13.png)

2) the second is that we increase the minimum number of samples at internal nodes (not
leaves)

![](/screenshots/s14.png)

3) the third way is to increase the minimum number of samples at leaf nodes

![](/screenshots/s15.png)

We can also combine the three conditions to perform more precise pruning.

### 2 Classification tree

So far we have discussed regression trees i.e. we are trying to predict a
continuous quantitative variable like the number of video game sales we will now discuss
classification trees in which will attempt to predict the categorical variable such as, no one has had
or not a heart attack.
First difference between the regression trees and the classification trees, we find the mean of the
variable to get the predicted variable, but for classification trees we will use the mode,
i.e. we assign this class to the region which is the most common class in this region. the
regression trees and classification use recursive binary splitting.

#### 2.1 Implementing a regression tree
For the classification tree, we chose a dataset that contains different data
of heart attack (age, sex of the person, already married, heart disease, fat level, status
smoking... etc.) and it predicts whether or not they had a stroke before.
34We will not return to the preprocessing this time for length reasons (everything will be in the
scripts provided).
##### Prediction and performance

![](/screenshots/s16.png)

in the regression tree, we used r2 and mse to calculate the performance of our model,
for classification models we can use precision_score and confusion matrix
precision_score is the percentage of records that we can correctly identify using
our model every time we run the confusion matrix we get a 2*2la matrix
first cell [0]0] represents the true negative, which means the predicted value of no stroke
(0) was correctly predicted the second cell [1]0] represents a false negative, which means that the
predicted value of no stroke (0) was incorrectly predicted the third cell [1]0] represents a false positive,
which means that the predicted value of stroke (1) was incorrectly predicted the third cell [1]0] represents a
true positive, which means that the predicted value of AVC(1) was correctly predictednote that we
can calculate the precision_score from the matrix, we divide the (true negative + true positive) by the
matrix sum for example here we have (479+0) / (479+2+30+0) which gives us a score of 0.93.

#### 2.2 Display the tree

![](/screenshots/s17.png)

Here each cell contains four values first contains the condition then we have the gini value
for this bucket, then we have the sample size of each node.
Note that the sum of all sample values of nodes in a certain level will be equal to the
sample value in root node and value shows bucket values so we have 4380 and
219 1 in the root node for example.
As for the color orange represents a high purity of 0 and blue represents a high purity of 1

#### 2.3 Conclusion on trees

Decision trees for regression and classification have a number of advantages over
compared to classical approaches such as regression such as:
1) trees are very easy to explain to people
2) trees reflect human decision-making (rationality) more closely than other approaches to
regression and classification
3) they can be displayed graphically, and are easily interpreted even by non-experts
For disadvantages, there is 1 major disadvantage, a simple decision tree usually does not have the
same predictive accuracy as some of the other regression and classification approaches but in
aggregating (combining) many decision trees or what we call creating sets,
we can significantly improve the performance of decision trees.

#### 2.4 Set Learning

The problem with decision trees is that decision trees have high variance, which
means if I have a dataset and I split the dataset into two parts and I use
each part to train a model, the two models will be very different in general.
So a natural way to reduce the variance and therefore increase the accuracy of the prediction is to
take many population training sets and then build a separate model by
using each training set, finally we average the resulting prediction to
get the final prediction.

#### 2.4.1 Bootstrapping

Let us understand how bootsrapping helps us to create different samples using the
same sample set suppose i have a data set of 5 numbers
7 9 5 4 3 I want to create 3 training sets from this training set. the method
is to randomly choose a number from this training set and add it to my set
training so that we have
9 5 4 3 4
7 9 5 4 7
8 9 9 4 3

#### 2.4.2 Bagging

1. During bagging, pruning is not carried out, full-length trees are grown.
2. Individual trees have high variance and low bias, mean reduces variance.
3. in regression, we take the mean of the predicted values.
374. in the classification, we take the majority vote, i.e. the most predicted value will be
considered the final prediction.

#### 2.4.3 Boosting

Now we are going to discuss Boosting, in Boosting we create a number of trees but the
difference is that the trees are grown sequentially, that means each tree is grown
using information from already grown trees, we will discuss 2 strengthening techniques,
gradient boosting and Ada boosting

#### 2.4.4 Gradient boosting

Gradient boosting is a slow learning procedure that involves adapting our tree using
the current residuals rather than the obtained result answer first we form the tree then we find the
difference between predicted and actual, these are called residuals Then we use these residuals to fit
a small shaft, but note that we control the length by boosting unlike bagging where
we create full length shafts. So this little tree, which is adjusted on the residuals is applied
with an indent parameter then is appended to the original tree so the second tree is essentially the
sum of the first tree and the newly created tree multiplied by a shrink parameter
now using the second tree we again find the residuals and again using the
residues, we fit again a small tree on these residues this small tree is again multiplied in
lambda and so on, so instead of learning by building the whole tree at once, amplifying
of the gradient learns slowly by building one small tree at a time, that's why it's called a slow learner

![](/screenshots/s18.png)

#### 2.4.5 Boosting Ada
ada boosting or adaptive boosting in this first, we create a tree, we find out the predictions
using this tree, wherever this tree has been misclassified or wherever the residue of this tree is very
large, we increase the importance of that particular observation and then we create a tree again
on our observations now this time since we have increased the importance of these observations
our tree will try to correctly capture or classify these observations so that this time we
get a second tree which will be a bit different from the first tree again we will find the
residuals or misclassified observations, we will increase the weight of these observations and run the
model again, this way we will continue to build the model for a number of
of predetermined time.

![](/screenshots/s19.png)

### 2.5 Naive Bayes
Here are some notations
P(A) means the probability of A, P(A/B) means the probability of A given that B has already happened
Bayes theory says that P (A/B) = P (B/A) * P(A) / P(B)
Returning now to our heart attack example, we will use the naive Bayes model to
predict the likelihood of a person having a heart attack given all the attributes that
look like P (heart attack / gender & hypertension & age & glucose level & .. ..)
Now the reason we call it naive is that we make the naive assumption that
variables are independent.
Two events A and B are independent if P (A/B) = P(A) and P (B/A) = P(B)

##### Naive Bayes
![](/screenshots/s20.png)

## Conclusion

To successfully create a good machine learning model, two fundamental things are required:
the first is to understand how the chosen algorithm works but it is not necessary to go into advanced mathematical details
the second thing is to have a good knowledge of data preprocessing.

