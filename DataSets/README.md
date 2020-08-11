Each folder contains the following files:

"dim.txt"
"bags.txt"
"aIndx.txt"
"aJndx.txt"
"aVal.txt"
"bagLabels.txt"
"trainFold.txt"
"testFold.txt"

Notation:
n  denote the number of feature (size of the sample space)
N  denote the number of instances
A  denote an (n x N) feature-instance matrix
nZ denote the number of nonzero components in A
nB denote the number of bags

File contents:

"dim.txt"                 contains (n, N, nZ, nB)
"bags.txt"                contains the assignment instance-to-bag
"bagLabels.txt"           contains the positive (+1) / negative (-1) label of each bag containing the instance
"aIndx.txt", "aJindx.txt" contain the pair i,j of each nonzero component of A
"aVal.txt"                contains the nZ nonzero component of A, i.e., for k in (1..nZ): A[aIndx[k]][aJindx[k]] = aVal[k]
"trainFold.txt"           contains 10 list of bag index, each containing (0.9 x nB) bag indexes, for the tenfold cross validation training.
"testFold.txt"            contains 10 list of bag index, each containing (0.1 x nB) bag indexes, for the tenfold cross validation testing.
