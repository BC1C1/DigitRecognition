#pragma once

#include "TypeDefine.h"
#include "SuperParam.h"
#include <qdir.h>
#include <qfile.h>
#include <iostream>

class DataReader_CIFAR {
public:
	void readAll() {
		readTrain();
		readTest();
	}
	void readTrain();
	void readTest();

	Matrix trainData;
	Matrix trainLabel;
	Matrix testData;
	Matrix testLabel;
};