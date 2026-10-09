find_package(Qt5 REQUIRED COMPONENTS Core)

add_executable(MachineLearning
Layers/Conv2D.cpp
Layers/Conv2D.h
Layers/Layer.cpp
Layers/Layer.h
Layers/Linear.cpp
Layers/Linear.h
Layers/MaxPool2D.cpp
Layers/MaxPool2D.h
Layers/ReLU.cpp
Layers/ReLU.h
LossFunctions/CrossEntropyLoss.cpp
LossFunctions/CrossEntropyLoss.h
LossFunctions/LossFunction.cpp
LossFunctions/LossFunction.h
LossFunctions/MSELoss.cpp
LossFunctions/MSELoss.h
Optimizer/Optimizer.cpp
Optimizer/Optimizer.h
Optimizer/SGD.cpp
Optimizer/SGD.h
Model/Model.cpp
Model/Model.h
config/SuperParam.h
Dao/DataReader.cpp
Dao/DataReader.h
utils/Utils.cpp
utils/Utils.h
src/main.cpp
)

target_include_directories(MachineLearning PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Layers
    ${CMAKE_CURRENT_SOURCE_DIR}/LossFunctions
    ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/Model
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_SOURCE_DIR}/Dao
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(MachineLearning PRIVATE Matrix)
target_link_libraries(MachineLearning PRIVATE Qt5::Core)