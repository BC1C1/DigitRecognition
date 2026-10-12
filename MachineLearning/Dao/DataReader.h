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
	// 传「数据集目录 + 用哪一份」而不是完整路径：四个文件名是 MNIST / Fashion-MNIST
	// 共有的约定，由这里自己拼，换数据集只要换 cfg::datasetDir 一处。
	//
	// ⚠ split 必须有：训练集是 train-*，测试集是 t10k-*。不区分就会**两份都读训练集**
	//   ——文件存在、magic 合法、N 还在文件范围内（10000 < 60000 不触发裁剪），
	//   于是测试集评估静默变成"训练集前 10000 张"。
	// @param split - "train" 或 "t10k"
	Matrix readPicture(size_t N, const QString& dir, const char* split);
	Matrix readLabel(size_t N, const QString& dir, const char* split);
private:
	size_t imageNumCache = 0;
};

