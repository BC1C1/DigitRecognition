#include "DataReader_CIFAR.h"

void DataReader_CIFAR::readTrain()
{
	const size_t batchSize = 10000;
	const size_t bodySize = 3 * 32 * 32;
	trainData = Matrix(batchSize * 5, bodySize);
	trainLabel = Matrix(batchSize * 5, 10);
	float* trainLabelRawData = new float[5 * batchSize * 10]();
	float* trainBodyRowData = new float[5 * batchSize * bodySize];
	QString rootDir = QString::fromUtf8(PROJECT_ROOT_DIR);
    QString dataDir = QDir(rootDir).filePath(cfg::cifarDatasetDir);
    for (size_t batch = 1; batch <= 5; batch++) {
        QString fileName = QString("data_batch_%1.bin").arg(batch);
        QString path = QDir(dataDir).filePath(fileName);

        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            std::cout << "cannot open " << path.toStdString() << std::endl;
            continue;
        }
        QByteArray array = file.readAll();
        file.close();
        qint64 expectedSize = (qint64)batchSize * (bodySize + 1);
        if (array.size() != expectedSize) {
            std::cout << "bad size: " << array.size()
                << ", expected: " << expectedSize << std::endl;
            continue;
        }
        size_t sampleBase = (batch - 1) * batchSize;

        for (size_t n = 0; n < batchSize; n++) {
            size_t recordOffset = n * (bodySize + 1);
            size_t globalSample = sampleBase + n;

            // label
            int l = static_cast<unsigned char>(array[(int)recordOffset]);
            trainLabelRawData[globalSample * 10 + l] = 1.0f;

            // body
            for (size_t i = 0; i < bodySize; i++) {
                size_t offset = recordOffset + 1 + i;
                float d = static_cast<float>(static_cast<unsigned char>(array[(int)offset]));
                trainBodyRowData[globalSample * bodySize + i] = d / 255.0f;
            }
        }
    }
	cudaMemcpy(trainData.data_gpu(), trainBodyRowData, sizeof(float) * bodySize * batchSize * 5, cudaMemcpyHostToDevice);
	cudaMemcpy(trainLabel.data_gpu(), trainLabelRawData, sizeof(float) * 10 * batchSize * 5, cudaMemcpyHostToDevice);
	delete[] trainLabelRawData; delete[] trainBodyRowData;
    std::cout << "data for train is ready" << std::endl;
}

void DataReader_CIFAR::readTest()
{
    const size_t batchSize = 10000;
    const size_t bodySize = 3 * 32 * 32;

    testData = Matrix(batchSize, bodySize);
    testLabel = Matrix(batchSize, 10);
    QString rootDir = QString::fromUtf8(PROJECT_ROOT_DIR);
    QString dataDir = QDir(rootDir).filePath(cfg::cifarDatasetDir);
    QString path = QDir(dataDir).filePath("test_batch.bin");
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        std::cout << "cannot open " << path.toStdString() << std::endl;
        return;
    }
    QByteArray array = file.readAll();
    file.close();

    qint64 expectedSize = (qint64)batchSize * (bodySize + 1);
    if (array.size() != expectedSize) {
        std::cout << "bad size: " << array.size()
            << ", expected: " << expectedSize << std::endl;
        return;
    }

    float* labelRaw = new float[batchSize * 10]();
    float* bodyRaw = new float[batchSize * bodySize];

    for (size_t n = 0; n < batchSize; n++) {
        size_t recordOffset = n * (bodySize + 1);
        int l = static_cast<unsigned char>(array[(int)recordOffset]);
        labelRaw[n * 10 + l] = 1.0f;

        for (size_t i = 0; i < bodySize; i++) {
            size_t offset = recordOffset + 1 + i;
            float d = static_cast<float>(
                static_cast<unsigned char>(array[(int)offset]));
            bodyRaw[n * bodySize + i] = d / 255.0f;
        }
    }

    cudaMemcpy(testData.data_gpu(), bodyRaw,
        sizeof(float) * batchSize * bodySize,
        cudaMemcpyHostToDevice);
    cudaMemcpy(testLabel.data_gpu(), labelRaw,
        sizeof(float) * batchSize * 10,
        cudaMemcpyHostToDevice);

    delete[] labelRaw;
    delete[] bodyRaw;
    std::cout << "data for test is ready" << std::endl;
}
