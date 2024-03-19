from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

# Path types
FileName = str
FilePath = str
DirName = str
DirPath = str

# Numpy types
f64 = np.float64
i64 = np.int64
ui8 = np.uint8
NpArr = npt.NDArray
NpArrf64 = npt.NDArray[f64]

# Numpy types.
NpVectorNf64 = Annotated[npt.NDArray[f64], Literal["N"]]
NpVector1f64 = Annotated[npt.NDArray[f64], Literal["1"]]
NpVector2f64 = Annotated[npt.NDArray[f64], Literal["2"]]
NpVector3f64 = Annotated[npt.NDArray[f64], Literal["3"]]
NpArrayN2f64 = Annotated[npt.NDArray[f64], Literal["N,2"]]
NpArrayN3f64 = Annotated[npt.NDArray[f64], Literal["N,3"]]
NpArrayNNf64 = Annotated[npt.NDArray[f64], Literal["N,N"]]
NpArrayMNf64 = Annotated[npt.NDArray[f64], Literal["M,N"]]

# Time stuff.
TimesVector = NpVectorNf64  # Time in seconds

# Control.
PositionsVector = NpVectorNf64
VelocitiesVector = NpVectorNf64
StateVector = NpVectorNf64
GainsVector = NpVectorNf64

# Geometry
PolygonArray = NpArrayN2f64
AnglesVector = NpVectorNf64

# Trajectories.
XYPath = NpArrayN2f64
XYThetaPath = NpArrayN3f64
