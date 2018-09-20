# DecisionTree
Python code for decision trees (ML class project)
Pip freeze for the codebase:

colorama==0.3.9
pandas==0.23.4
sklearn==0.0
termcolor==1.1.0

Pruning Strategies:

First Strategy:

Prune the parent of leaf nodes.
 --If the parent is root (leaf id is 1 or 2), then don't prune it.
Prune 1/10 fraction of innerNodes.
 --Assumption no point pruning leaf node.
 --Shuffle ID's of inner nodes, and select 1/10 fraction of total inner node ID's randomly.
 -- If the randomly selected ID is either 0/1/2, remove it from the ids of nodes to be pruned and print a message.

 Command to run the code:
  -- python driver.py
  -- For now the data is stored locally and read from iris.csv (put it in the same folder as driver.py)
  -- To change the source of data make changes to the line:
    df = pd.read_csv('iris.csv', sep=',') #driver.py
  -- getLeafNodes/getInnerNodes require two parameters for the recursion to work as expected.
