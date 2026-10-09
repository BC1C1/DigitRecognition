#include "DataReader.h"


Matrix DataReader::readPicture(size_t N, QString filename) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "cannot open" << filename;
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
    const char* base = allPix.constData();
    for (size_t i = 0; i < N; i++) {
        const char* row = base + i * rows * cols;
        for (size_t j = 0; j < (size_t)(rows * cols); j++) {
            ret(i, j) = double(static_cast<unsigned char>(row[j])) / 255.0;
        }
    }

    return ret;
}

Matrix DataReader::readLabel(size_t N, QString filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug() << "cannot open" << filename;
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

    for (size_t i = 0; i < N; ++i)
    {
        unsigned char lab = base[i];

        ret(i, lab) = 1.0;
    }

    return ret;
}

