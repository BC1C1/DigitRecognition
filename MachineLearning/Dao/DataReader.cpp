#include "DataReader.h"
#include <qdir.h>
#include <iostream>
namespace {

// 同一个逻辑文件的两种命名约定：
//   MNIST          train-images.idx3-ubyte   （点号）
//   Fashion-MNIST  train-images-idx3-ubyte   （连字符）
// 两者只差 idx3 前面那一个字符，但 QFile 找不到时不会告诉你拼错了，
// 所以这里不猜、也不硬编码，直接把两种都试一遍。
//
// ⚠ operand 是**不带后缀**的前半段（"train-images"），后缀由这里自己加。
//   之前把前半段写成带点的 "train-images."，于是连字符版本拼成了
//   "train-images.-idx3-ubyte" —— 畸形名，两边都找不到，还静默走了兜底。
//
// 两个形参都用 QString：调用点会传 QString（QString::fromLatin1(split) + "-images"），
// 传 const char* 的话是 QString→const char*，转不过去（C2664）。
QString findDataFile(const QString& dir, const QString& operand, const QString& suffix)
{
	const QString base = QDir(dir).filePath(operand);
	const QString dotted = base + QLatin1Char('.') + suffix;   // train-images.idx3-ubyte
	const QString dashed = base + QLatin1Char('-') + suffix;   // train-images-idx3-ubyte
	if (QFile::exists(dotted)) return dotted;
	if (QFile::exists(dashed)) return dashed;
	return dotted;   // 都不存在时返回点号名，让调用方的报错里出现一个可读的路径
}

} // namespace


Matrix DataReader::readPicture(size_t N, const QString& dir, const char* split) {
    // split 决定用哪一份：train-* 还是 t10k-*。写死成 "train" 会让测试集评估
    // 静默读成训练集的前 N 张（文件合法、N 在范围内，所以什么都不报）。
    const QString path = findDataFile(dir, QString::fromLatin1(split) + "-images", "idx3-ubyte");
    std::cout << "now reading images..." << std::endl;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "cannot open" << path;
        return Matrix();
    }

    // 读头
    QByteArray head = file.read(16);
    if (head.size() != 16) {
        qDebug() << "cannot read header";
        return Matrix();
    }

    qint32 magic = qFromBigEndian<qint32>((const uchar*)head.data());
    qint32 numImgs = qFromBigEndian<qint32>((const uchar*)head.data() + 4);
    qint32 rows = qFromBigEndian<qint32>((const uchar*)head.data() + 8);
    qint32 cols = qFromBigEndian<qint32>((const uchar*)head.data() + 12);

    // 校验
    if (magic != 0x00000803) {
        qDebug() << "bad magic:" << magic;
        return Matrix();
    }
    if (rows != 28 || cols != 28) {
        qDebug() << "unexpected size:" << rows << cols;
        return Matrix();
    }
    if (N > (size_t)numImgs) {
        qDebug() << "N > numImgs, clipping:" << N << "->" << numImgs;
        N = numImgs;
    }

    QByteArray allPix = file.read(N * rows * cols);
    if (allPix.size() != (int)(N * rows * cols)) {
        qDebug() << "pixel data incomplete:" << allPix.size()
            << "expected" << N * rows * cols;
        return Matrix();
    }

    Matrix ret({ N, (size_t)(rows * cols) });
    float* cpuData = new float[N * rows * cols];
    const char* base = allPix.constData();
    for (size_t i = 0; i < N; i++) {
        const char* row = base + i * rows * cols;
        for (size_t j = 0; j < (size_t)(rows * cols); j++) {
            cpuData[i * rows * cols + j] = static_cast<float>(static_cast<unsigned char>(row[j])) / 255.0f;
        }
    }
    cudaMemcpy(ret.data_gpu(), cpuData, sizeof(float) * ret.size(), cudaMemcpyHostToDevice);
    delete[] cpuData;
    std::cout << "read success! filepath: " << path.toStdString() << std::endl;
    return ret;
}

Matrix DataReader::readLabel(size_t N, const QString& dir, const char* split)
{
    const QString path = findDataFile(dir, QString::fromLatin1(split) + "-labels", "idx1-ubyte");
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug() << "cannot open" << path;
        return Matrix();
    }

    // 标签文件头固定8字节
    QByteArray head = file.read(8);
    if (head.size() != 8)
    {
        qDebug() << "cannot read label header";
        return Matrix();
    }

    qint32 magic = qFromBigEndian<qint32>((const uchar*)head.data());
    qint32 numLabels = qFromBigEndian<qint32>((const uchar*)head.data() + 4);

    // idx1‑ubyte魔数 0x00000801
    if (magic != 0x00000801)
    {
        qDebug() << "bad label magic:" << magic;
        return Matrix();
    }

    if (N > (size_t)numLabels)
    {
        qDebug() << "N > numLabels, clipping:" << N << "->" << numLabels;
        N = numLabels;
    }

    QByteArray labelBuf = file.read(N);
    if (labelBuf.size() != (int)N)
    {
        qDebug() << "label data incomplete:" << labelBuf.size() << "expected" << N;
        return Matrix();
    }

    // 输出 N*10
    Matrix ret({ N, 10 });
    const unsigned char* base = reinterpret_cast<const unsigned char*>(labelBuf.constData());
    float* cpuData = new float[N * 10]();   // 值初始化：只写 one-hot 的一位，其余必须确定是 0
    for (size_t i = 0; i < N; ++i)
    {
        unsigned char lab = base[i];
        cpuData[i * 10 + lab] = 1.0;
    }
    cudaMemcpy(ret.data_gpu(), cpuData, sizeof(float) * ret.size(), cudaMemcpyHostToDevice);
    delete[] cpuData;
    return ret;
}

