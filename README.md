# 2. Compile K-Means approach code

## Set Enviroment and Loading data


```python
import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.clustering import KMeans, KMeansModel

def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print ("File %s does not exist." % ratingsFile)
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print ("No ratings provided.")
        sys.exit(1)
    else:
        return ratings
```


```python
# set up environment
conf = SparkConf() \
    .setAppName("MovieKMeans") \
    .set("spark.executor.memory", "6g")
sc = SparkContext(conf=conf)
f= open("/vagrant/files/HW04/kmeans_output.txt","w")

# load personal ratings
myRatings = loadRatings(sys.argv[2])
myRatingsRDD = sc.parallelize(myRatings, 1)

# load ratings and movie titles

movieLensHomeDir = sys.argv[1]

# ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)
movies = sc.textFile(join(movieLensHomeDir, "movies.dat")).cache()


numRatings = ratings.count()
numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

# split into training and test dataset
numPartitions = 4
training = ratings.filter(lambda x: x[0] < 8) \
    .values() \
    .union(myRatingsRDD) \
    .repartition(numPartitions) \
    .cache()
test = ratings.filter(lambda x: x[0] >= 8).values().cache()

numTraining = training.count()
numTest = test.count()
```

## get movie types


```python
def parseMovieType(line):
    typePart= (line.strip().split("::"))[2].strip().split("|")
    for i in typePart:
        if i not in types:
            types.append(i)
    return types


#movieTypes=[]
types=[]
collectTypes= sc.textFile("file:///vagrant/files/HW04/movies.dat").map(parseMovieType).collect()
movieTypes = collectTypes[-1]
print (movieTypes)


```

    [u'Animation', u"Children's", u'Comedy', u'Adventure', u'Fantasy', u'Romance', u'Drama', u'Action', u'Crime', u'Thriller', u'Horror', u'Sci-Fi', u'Documentary', u'War', u'Musical', u'Mystery', u'Film-Noir', u'Western']


## Add Type colunms 


```python
def addMovieTypes(line):
    fields=line.strip().split("::")
    typePart =fields[2].strip().split("|")
    for i in range(typeNum):
        if movieTypes[i] in typePart:
            fields.append(1)
        else:
            fields.append(0)
    fields[0]= int(fields[0])
    fields.pop(2)
    return fields
    #int(fields[0]), fields[1], fiedls[3:-1]
            
typeNum=len(movieTypes)
movieComplete= np.array (sc.textFile("file:///vagrant/files/HW04/movies.dat").map(addMovieTypes).collect())
print(movieComplete)

movieTypeInfo= movieComplete[:,2:-1].astype(np.int)
movieTypeInfo=sc.parallelize(movieTypeInfo)
```

    [u'Animation', u"Children's", u'Comedy', u'Adventure', u'Fantasy', u'Romance', u'Drama', u'Action', u'Crime', u'Thriller', u'Horror', u'Sci-Fi', u'Documentary', u'War', u'Musical', u'Mystery', u'Film-Noir', u'Western']
    [[u'1' u'Toy Story (1995)' u'1' ... u'0' u'0' u'0']
     [u'2' u'Jumanji (1995)' u'0' ... u'0' u'0' u'0']
     [u'3' u'Grumpier Old Men (1995)' u'0' ... u'0' u'0' u'0']
     ...
     [u'3950' u'Tigerland (2000)' u'0' ... u'0' u'0' u'0']
     [u'3951' u'Two Family House (2000)' u'0' ... u'0' u'0' u'0']
     [u'3952' u'Contender, The (2000)' u'0' ... u'0' u'0' u'0']]


## get movie clusters Model


```python
# get movie clusters Model
def clusterMovies(movieTypeInfo):
    i = 2
    
    def getWcv(point,kModel):
        center = kModel.centers[kModel.predict(point)]
        return sum([x**2 for x in (point - center)])
    
    def getBcv(point, kModel ):
        center = kModel.centers[kModel.predict(point)]
        return sum([x**2 for x in (center - xAve)])
            
    
    pointsNum = len(movieComplete)
    xAve=( sum(movieTypeInfo.collect()) .astype(np.float))/pointsNum
    bestCH= 0
    bestKModel= None
    K=i
    records=[]
    while i<=20:
        movieCluster = KMeans.train(movieTypeInfo, i, maxIterations=100, initializationMode="random")
        
        # Due to the Within Cluster Variation would always decrease while K increase
        # WCV is not a good criteria to judge the performance of K-Means Cluster
        # Therefore, I choose to use the the CH Index to find the best K
        wcv= movieTypeInfo.map(lambda point: getWcv(point,movieCluster)).reduce(lambda x, y: x + y)
        bcv= movieTypeInfo.map(lambda point: getBcv(point,movieCluster)).reduce(lambda x, y: x + y)
        chIndex = ( bcv/(i-1) )/ (wcv/(pointsNum-i))
        
        # Find the highest CH Index
        if bestCH< chIndex:
            bestKModel = movieCluster
            bestCH=chIndex
            K=i
        records.append((chIndex, wcv,bcv))
        f.write("When K = {}, the value of CH index is {}.\n".format(i, chIndex))

        i += 1
    return  bestKModel,K,chIndex,records
        
movieCluster = clusterMovies(movieTypeInfo)
kMeansModel= movieCluster[0]
```



## combine movie with Cluster


```python
length = len(tempClusters)
movieWithCluster=[]
for i in range(length):
    temp=[int(movieComplete[i,0]), movieComplete[i,1], tempClusters[i]]
    movieWithCluster.append(temp)
# A Matrix of [movieID, movieTitle, movieCluster]
movieWithCluster= np.array(movieWithCluster)
print (movieWithCluster[0:10])
```

    [[u'1' u'Toy Story (1995)' u'4']
     [u'2' u'Jumanji (1995)' u'0']
     [u'3' u'Grumpier Old Men (1995)' u'4']
     [u'4' u'Waiting to Exhale (1995)' u'1']
     [u'5' u'Father of the Bride Part II (1995)' u'4']
     [u'6' u'Heat (1995)' u'3']
     [u'7' u'Sabrina (1995)' u'4']
     [u'8' u'Tom and Huck (1995)' u'0']
     [u'9' u'Sudden Death (1995)' u'3']
     [u'10' u'GoldenEye (1995)' u'3']]


## get cluster num to each rating row


```python
def getUserClusterRating(line):
    userID = int( line[0])
    movieID = str(line[1])
    movieScore = line[2]
    tempIndex = np.argwhere(movieWithCluster[:,0]==movieID)[0][0]
    clusterInd = int( movieWithCluster[tempIndex, 2])
    return (userID*100+clusterInd, movieScore)
userClusterRatings = training.map(getUserClusterRating).groupByKey().mapValues(lambda x: sum(x)/len(x)).collect()
tempList=[]
for cluRating in userClusterRatings:
    userID = cluRating[0]//100
    clusterInd = cluRating[0]%100
    score = round(cluRating[1],4)
    tempList.append([userID, clusterInd,  score])

# A matrix [userID,  clusterID, clusterAveScore ]
userClusterRatings= np.array(tempList).astype(np.float)
```


```python
userClusterRatings
```




    array([[  0.    ,   0.    ,   3.8571],
           [215.    ,   4.    ,   5.    ],
           [  0.    ,   4.    ,   3.2   ],
           ...,
           [336.    ,   3.    ,   4.2857],
           [ 46.    ,   3.    ,   1.    ],
           [ 87.    ,   3.    ,   3.125 ]])



## return the selected user's cluster score


```python
def getClusterAveRating(userID):
    userIndex = np.argwhere(userClusterRatings[:,0]==userID)
    clusterAveRating=[]
    for i in userIndex:
        clusterAveRating.append(userClusterRatings[i[0],1:])
    return clusterAveRating

# here got the info for user 0
clusterAveRating = np.array(getClusterAveRating(0))
```


```python
clusterAveRating
```




    array([[0.    , 3.8571],
           [4.    , 3.2   ],
           [1.    , 4.    ],
           [3.    , 1.    ]])



## Define  a function to compute the score


```python

def getScore(movieID):
    movieIndex= np.argwhere(movieWithCluster[:,0]==str(movieID))[0][0]
    
    clusterNum = movieWithCluster[movieIndex, 2]
    clusterNum = int(clusterNum) 
    if clusterNum not in clusterAveRating[:,0]:
        score =0 
    else:
        clusterIndex = np.argwhere(clusterAveRating[:,0]== clusterNum)[0][0]
        score = clusterAveRating[clusterIndex,1]
    return score
```

## Compute the RMSE for K-Means model


```python
def computeKmeansRmse(line):
    userID = int( line[0])
    movieID = str(line[1])
    movieScore = line[2]
    clusterAveRating = getClusterAveRating(userID)
    predictScore = getScore(movieID)
    error =(predictScore-movieScore)**2
    return error

kmeansRmse = sqrt ( test.map(computeKmeansRmse).reduce(add) / numTest)
rount(kmeansRmse,4)
```




    2.0095



## Recommandation


```python
moviesDict = dict(movies.map(parseMovie).collect())
myRatedMovieIds = set([x[1] for x in myRatings])
candidates = sc.parallelize([m for m in moviesDict if m not in myRatedMovieIds])

predictScores = candidates.map(getScore).collect()
predictions= np.c_[candidates.collect(), predictScores]
recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:50]
for i in xrange(len(recommendations[0:10])):
    f.write ( ("%2d: %s\n" % (i + 1, moviesDict[recommendations[i][0]])).encode('ascii', 'ignore') )
```

    [[u'1' u'Toy Story (1995)' u'4']
     [u'2' u'Jumanji (1995)' u'0']
     [u'3' u'Grumpier Old Men (1995)' u'4']
     [u'4' u'Waiting to Exhale (1995)' u'1']
     [u'5' u'Father of the Bride Part II (1995)' u'4']
     [u'6' u'Heat (1995)' u'3']
     [u'7' u'Sabrina (1995)' u'4']
     [u'8' u'Tom and Huck (1995)' u'0']
     [u'9' u'Sudden Death (1995)' u'3']
     [u'10' u'GoldenEye (1995)' u'3']]
