from colorama import init 
from DecisionTree import *
from termcolor import colored
import pandas as pd
from sklearn import model_selection
import random
import copy

#Function returns node(ID's) to pruned for our two pruning strategies.
def selectNodesToPrune():
	pruningStratergy = {
		'Prune nodes one level above leaf' : pruneNodesAboveLeaf(),
		'Randomly select n nodes and prune' : pruneRandomNodes()
	}
	return pruningStratergy
#Function returns node(ID's) to pruned for pruning strategy - Prune nodes one level above leaf
def pruneNodesAboveLeaf() :
	idOfNodesAboveLeaf = []
	leaves = getLeafNodes(t_trained, [])
	#If left/right child of root is a leaf node, do not prune the root!! Skip the root!
	for leaf in leaves:
		if (leaf.id!=1 and leaf.id!=2):
			if(leaf.id%2):
				#parentID when the child is left-child
				parentId = (leaf.id-1)/2
			else:
				#parentID when the child is right-child
				parentId = (leaf.id-2)/2
			#prevent duplicate id of parent
			if parentId not in idOfNodesAboveLeaf:  
				idOfNodesAboveLeaf.append(parentId)
		else:
			#if lead id is 1/2 skip
			continue;
	return idOfNodesAboveLeaf
#Function returns node(ID's) to pruned for pruning strategy - Randomly select n nodes and prune
def pruneRandomNodes():
	innernodeIdList = []
	totalNodes = 0;
	inneNodes = getInnerNodes(t_trained, [])
	for inner in innerNodes:
		innernodeIdList.append(inner.id)
	random.shuffle(innernodeIdList)
	#Take 1/10th of the total internal node ids
	#Change variable 0.1 to prune different fraction of node. For example to prune 20% of internal node, change 0.1 to 0.2.
	numOfNodesToPrune = int(math.ceil(0.1*(len(innernodeIdList))))
	listOfPruneNodeId = innernodeIdList[:numOfNodesToPrune] 
	#Do not prune node with IDs 0, 1, 2
	for ele in [0,1,2]:
		if ele in listOfPruneNodeId:
			listOfPruneNodeId.remove(ele)
			print(colored('Selected node was : ID ' + str(ele) + '!! Thus removed it!!'), 'red')
	return listOfPruneNodeId

header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
#df = pd.read_csv('https://gist.github.com/curran/a08a1080b88344b0c8a7#file-iris-csv', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
#df = pd.read_csv('https://gist.github.com/curran/a08a1080b88344b0c8a7#file-iris-csv', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
df = pd.read_csv('iris.csv', sep=',')
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)
print("********** Leaf nodes ****************")
leaves = getLeafNodes(t, [])
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t, [])
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
#0.2 is the current test ratio of the total data available. To train model on 0.5/0.5 test/train ratio, change 0.2 to 0.5.
trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()
t_trained = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t_trained)
acc = computeAccuracy(test, t_trained)
print("Accuracy on test = " + str(acc))

pruningStrategyDict = selectNodesToPrune()
for strategy in pruningStrategyDict:
	print(colored("************* Pruning Strategy : " + strategy + "*******", 'green'))
	local_t_trained = copy.deepcopy(t_trained)
	## TODO: You have to decide on a pruning strategy
	t_pruned = prune_tree(local_t_trained, pruningStrategyDict[strategy])
	print("*************Tree after pruning*******")
	print_tree(t_pruned)
	acc = computeAccuracy(test, t_pruned)
	print("Accuracy on test = " + str(acc))

