"""Default RDKit normalization transformtions

taken from: Code/GraphMol/MolStandardize/TransformCatalog/normalizations.in
commit fbad8f73 2024-06-03
"""

__all__ = ["standard"]


standard = """\
Nitro to N+(O-)=O\t[N,P,As,Sb;X3:1](=[O,S,Se,Te:2])=[O,S,Se,Te:3]>>[*+1:1]([*-1:2])=[*:3]
Sulfone to S(=O)(=O)\t[S+2:1]([O-:2])([O-:3])>>[S+0:1](=[O-0:2])(=[O-0:3])
Pyridine oxide to n+O-\t[nH0+0:1]=[OH0+0:2]>>[n+:1][O-:2]
Azide to N=N+=N-\t[*:1][N:2]=[N:3]#[N:4]>>[*:1][N:2]=[N+:3]=[N-:4]
Diazo/azo to =N+=N-\t[*:1]=[N:2]#[N:3]>>[*:1]=[N+:2]=[N-:3]
Sulfoxide to -S+(O-)-\t[!O:1][S+0;X3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]
Phosphate to P(O-)=O\t[O,S,Se,Te;-1:1][P+;D4:2][O,S,Se,Te;-1:3]>>[*+0:1]=[P+0;D5:2][*-1:3]
C/S+N to C/S=N+\t[C,S&!$([S+]-[O-]);X3+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]
P+N to P=N+\t[P;X4+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]
Normalize hydrazine-diazonium\t[CX4:1][NX3H:2]-[NX3H:3][CX4:4][NX2+:5]#[NX1:6]>>[CX4:1][NH0:2]=[NH+:3][C:4][N+0:5]=[NH:6]
Recombine 1,3-separated charges\t[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[N,P,As,Sb,O,S,Se,Te;+1:3]>>[*-0:1]=[*:2]-[*+0:3]
Recombine 1,3-separated charges\t[n,o,p,s;-1:1]:[a:2]=[N,O,P,S;+1:3]>>[*-0:1]:[*:2]-[*+0:3]
Recombine 1,3-separated charges\t[N,O,P,S;-1:1]-[a:2]:[n,o,p,s;+1:3]>>[*-0:1]=[*:2]:[*+0:3]
Recombine 1,5-separated charges\t[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[A:3]-[A:4]=[N,P,As,Sb,O,S,Se,Te;+1:5]>>[*-0:1]=[*:2]-[*:3]=[*:4]-[*+0:5]
Recombine 1,5-separated charges\t[n,o,p,s;-1:1]:[a:2]:[a:3]:[c:4]=[N,O,P,S;+1:5]>>[*-0:1]:[*:2]:[*:3]:[c:4]-[*+0:5]
Recombine 1,5-separated charges\t[N,O,P,S;-1:1]-[c:2]:[a:3]:[a:4]:[n,o,p,s;+1:5]>>[*-0:1]=[c:2]:[*:3]:[*:4]:[*+0:5]
Normalize 1,3 conjugated cation\t[N,O!$(*N);+0!H0:1]-[A:2]=[N!$(*~[N,O,P,S;-1]),O;+1H0:3]>>[*+1:1]=[*:2]-[*+0:3]
Normalize 1,3 conjugated cation\t[n;+0!H0:1]:[c:2]=[N!$(*~[N,O,P,S;-1]),O;+1H0:3]>>[*+1:1]:[*:2]-[*+0:3]
Normalize 1,5 conjugated cation\t[N,O!$(*N);+0!H0:1]-[A:2]=[A:3]-[A:4]=[N!$(*~[N,O,P,S;-1]),O;+1H0:5]>>[*+1:1]=[*:2]-[*:3]=[*:4]-[*+0:5]
Normalize 1,5 conjugated cation\t[n;+0!H0:1]:[a:2]:[a:3]:[c:4]=[N!$(*~[N,O,P,S;-1]),O;+1H0:5]>>[n+1:1]:[*:2]:[*:3]:[*:4]-[*+0:5]
Charge normalization\t[F,Cl,Br,I,At;-1:1]=[O:2]>>[*-0:1][O-:2]
Charge recombination\t[N,P,As,Sb;-1:1]=[C+;v3:2]>>[*+0:1]#[C+0:2]
"""
