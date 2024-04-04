from typing import Annotated, Any, Callable, Literal

import numpy as np
import numpy.typing as npt

# Path types
FileName = str
FilePath = str
DirName = str
DirPath = str

# Attrs stuff
AttrsConverterFunc = Callable[[Any], Any]
AttrsValidatorFunc = Callable

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
NpVector4f64 = Annotated[npt.NDArray[f64], Literal["4"]]
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
XYPoint = NpArrayN2f64
XYZPoint = NpArrayN3f64
PolygonArray = NpArrayN2f64
AnglesVector = NpVectorNf64

# Trajectories.
XYThetaPose = NpVector3f64
XYZThetaPose = NpVector4f64
XYPath = NpArrayN2f64
XYThetaPath = NpArrayN3f64
