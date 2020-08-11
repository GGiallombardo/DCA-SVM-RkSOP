Each dataset is denoted by two digits according to the following:

01 = Breast Cancer

02 = Diabetes

03 = Heart

04 = Ionosphere

05 = Brain_Tumor1

06 = Brain_Tumor2

07 = DLBCL

08 = Leukemia/ALLAML

For each dataset XY, the relevant data are grouped in 5 files:
"XY_dim.txt"
"XY_Indx.txt"
"XY_Jndx.txt"
"XY_aVal.txt"
"XY_Lab.txt"

NOTATION:

n  = number of features (size of the sample space)

N  = number of points

A  = (n x N) feature-point matrix

nZ = number of nonzero components in A


FILE CONTENTS

"XY_dim.txt"                  contains (n, N)

"XY_Lab.txt"                  contains the positive (+1) / negative (0) label of each point

"XY_Indx.txt", "XY_Jindx.txt" contain the pair i,j of each nonzero component of A

"XY_aVal.txt"                 contains the nZ nonzero component of A, i.e., for k in (1..nZ): A[aIndx[k]][aJindx[k]] = aVal[k]

