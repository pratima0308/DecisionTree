# DecisionTree
Python code for decision trees (ML class project)<br />
Pip freeze for the codebase:<br /><br />

colorama==0.3.9<br />
pandas==0.23.4<br />
sklearn==0.0<br />
termcolor==1.1.0<br />

Pruning Strategies:<br /><br />

First Strategy:<br />

Prune the parent of leaf nodes.<br />
 -- If the parent is root (leaf id is 1 or 2), then don't prune it.<br />
 -- Prune 1/10 fraction of innerNodes.<br />
 -- Assumption no point pruning leaf node.<br />
 -- Shuffle ID's of inner nodes, and select 1/10 fraction of total inner node ID's randomly.<br />
 -- If the randomly selected ID is either 0/1/2, remove it from the ids of nodes to be pruned and print a message.<br />

 Command to run the code:<br />
  -- python driver.py<br />
  -- For now the data is stored locally and read from iris.csv (put it in the same folder as driver.py)<br />
  -- To change the source of data make changes to the line:<br />
     df = pd.read_csv('iris.csv', sep=',') #driver.py<br />
  -- getLeafNodes/getInnerNodes require two parameters for the recursion to work as expected.<br />
