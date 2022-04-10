# Machine-Learning-Models
Ce readme est disponible en deux langues: [Anglais](#english) et [Français](#français)
This readme has two languages : [English](#english) and [French](#français)


## Français:


### Intro





## English:

### Intro

### Arbre de régression

Le premier ensemble de données est un ensemble de ventes de jeux vidéo et il a ces attributs

s1

Nous importons d'abord le fichier csv dans un data frame à l'aide de la bibliothèque pandas

s2

#### 1.2 Prétraitement des données

Maintenant nous commençons le prétraitement des données, d'abord nous identifions l'attribut que nous
voulons prédire, nous avons choisi les ventes de jeux vidéo européens, maintenant nous identifions les
attributs "inutiles" à notre module d’apprentissage, qui sont le nom, le rang et le global_sales (nous avons
supprimé le dernier car si nous le laissons le module peut simplement tout ignorer et faire une simple
soustraction à trouver les eu_sales)

s3


Maintenant que nous nous sommes débarrassés des attributs indésirables, nous devons examiner de plus
près les données et trouver des éléments de données manquants les attributs.

s4

Ici on peut voir qu'il manque une année de jeux vidéo et un éditeur. Nous avons deux manières de traiter ce
problème de manquant :
1) on remplace l'attribut manquant par la moyenne de toutes les données d'un même attribut (moyenne de
l'année) et pour l'éditeur manquant on peut le remplacer par l'éditeur le plus courant (en termes
d'occurrence), ainsi le module va ne pas être trop affecté lors de l'utilisation de la moyenne
2) la deuxième méthode consiste à supprimer les données avec des attributs manquants, nous utiliserons
cette méthode car les données manquantes sont si petites qu'elles sont fondamentalement négligeables (elles
représentent 0,01 pour cent de l'ensemble de données total)

s5


Maintenant que nous avons pris les données manquantes, l'étape suivante du prétraitement des données est
la transformation des attributs d'objet ou ce qu'on appelle la création de variables factices, nous pouvons
prendre un exemple d'attribut tel que Online_availablity qui prend soit "YES" soit "NO" pour la valeur,
nous pouvons le transformer en Online_availablity_YES qui prend 1 s'il est disponible en ligne et 0 s'il ne
l'est pas.
En général pour les attributs d'objet avec N valeurs différentes, nous les transformons en N-1 variables
fictives, dans notre exemple nous avons 3 "plateforme", "genre" et "Publisher"

s6


Maintenant avec cela fait, nous avons plus de 600 attributs.
28l'étape finale consiste à diviser le cadre de données en deux parties, nos variables x qui sont nos attributs
indépendants (année, éditeur, ...) et la variable y qui est notre variable cible ou variable dépendante (ventes
UE) et nous pouvons commencer l'entrainement.

s7

#### 1.3 L'entrainement de model

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

s8

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

s9


Nous devons d'abord importer tree depuis sklearn puis nous créons un objet regtree ici regtree est notre
nom de variable et nous utiliserons tree.DecisionTreeRegeressor() il y a plusieurs paramètres à l'intérieur,
et nous entraînons le regtree en utilisant la méthode fit.

s10


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

s11


#### 1.4 Afficher l'arbre 
La première étape consiste à créer un fichier de points, puis nous devons convertir ce fichier de points en
une image, puis nous utiliserons cette image pour créer un graphique comme on peut le voir , le graphe est
un arbre avec chaque nœud ayant : 1- le X[i] est la variable ieme de l'arbre et x[i]<554 par exemple est la
condition si la condition est vérifiée, nous irons au nœud de gauche, sinon nous irons au nœud de droite,
nous aurons ensuite le nombre d'échantillons dans le nœud. notez que la somme de toutes les valeurs
d'échantillon des nœuds dans un certain niveau sera égale à la valeur d'échantillon dans le nœud racine, et
nous avons la valeur moyenne des nœuds mse est l'erreur quadratique moyenne

s12


#### 1.5 Élagage des arbres

Maintenant les arbres de décision qui sont trop gros ont 2 problèmes, ils sont difficiles à interpréter et ils
surdimensionnent les données d'apprentissage donnant ainsi mauvaise performance de test, là pour nous
avons décidé de contrôler la croissance des arbres, l'élagage des arbres, dans cette stratégie nous dessinons
un très grand arbre puis taillez-le, ou vous pouvez dire que nous en coupons des sous-parties qui ne sont
pas bénéfiques afin d'obtenir un sous-arbre optimal.
il y a 3 façons
1) la première consiste à diminuer le nombre maximum de niveaux autorisés dans notre arbre

s13


2) la seconde est que nous augmentons le nombre minimum d'échantillons aux nœuds internes (non
feuilles)

s14

3) la troisième façon consiste à augmenter le nombre minimum d'échantillons aux nœuds feuilles

s15

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

s16


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

s17


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

s18

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

s19


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

s20

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

s21


3## Conclusion

Pour réussir à créer un bon modèle d’apprentissage automatique il a nécessité deux choses
fondamentales : la première est de bien comprendre comment l’algorithme choisi fonctionne
mais il n’est pas nécessaire de rentrer dans les détails mathématiques poussées, la deuxième
chose est d’avoir une bonne connaissance sur les prétraitements des données.

