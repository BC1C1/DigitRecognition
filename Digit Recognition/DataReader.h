#pragma once

#include <qstring.h>
#include "TypeDefine.h"
#include "qfile.h"
#include <QtEndian>

#include "qdebug.h"

class DataReader
{
public:
	/*bool checkFileHead();*/
	Matrix readPicture(size_t N, QString filename);
	Matrix readLabel(size_t N, QString filename);
private:
	size_t imageNumCache = 0;
};

