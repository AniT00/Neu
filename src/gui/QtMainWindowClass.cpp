#include "QtMainWindowClass.h"
#include <qlegendmarker.h>
// #include <QVboxlayout>

QAction* openDataset;
QtMainWindowClass::QtMainWindowClass(QWidget* parent)
  : QMainWindow(parent)
{
  ui.setupUi(this);

  m_neuralNetwork = nullptr;

  m_loss_series = new QLineSeries(this);
  m_accuracy_series = new QLineSeries(this);

  m_lbllearningRate = new QLabel("Learning rate:");
  m_boxLearningRate = new QDoubleSpinBox();
  m_boxLearningRate->setSingleStep(0.05);
  m_bttnTrain = new QPushButton("Train");
  m_verticalLayoutFill = new QWidget();

  m_verticalLayout = new QVBoxLayout();

  m_datasetLabel = new QLabel("Dataset file: ", this);
  m_training_chart_view = new QChartView(this);
  m_verticalLayout->addWidget(m_datasetLabel, 1);
  m_verticalLayout->addWidget(m_lbllearningRate);
  m_verticalLayout->addWidget(m_boxLearningRate);
  m_verticalLayout->addWidget(m_bttnTrain);
  m_verticalLayout->addWidget(m_verticalLayoutFill, 1);

  horizontalLayout = new QHBoxLayout(ui.centralWidget);
  horizontalLayout->setSizeConstraint(QLayout::SetMaximumSize);
  horizontalLayout->addLayout(m_verticalLayout, 1);
  horizontalLayout->addWidget(m_training_chart_view, 3);

  // m_training_chart_view->setSizePolicy(QSizePolicy::Preferred,
  //                                      QSizePolicy::Preferred);

  m_axisY = new QValueAxis();
  m_axisX = new QValueAxis();
  m_axisY->setLabelFormat("%.2f");
  m_axisY->setTitleText("Numero dispositivi");
  m_axisY->setMin(0);
  m_axisY->setMax(1);
  m_axisX->setLabelFormat("%.2f");
  m_axisX->setTitleText("Numero dispositivi");
  m_axisX->setMin(0);
  m_axisX->setMax(100);

  m_training_chart_view->chart()->addAxis(m_axisY, Qt::AlignLeft);
  m_training_chart_view->chart()->addAxis(m_axisX, Qt::AlignBottom);
  m_training_chart_view->chart()->addSeries(m_loss_series);
  m_training_chart_view->chart()->legend()->markers(m_loss_series)[0]->setLabel(
    "loss");
  m_loss_series->attachAxis(m_axisY);
  m_loss_series->attachAxis(m_axisX);
  m_training_chart_view->chart()->addSeries(m_accuracy_series);
  m_training_chart_view->chart()
    ->legend()
    ->markers(m_accuracy_series)[0]
    ->setLabel("accuracy");
  m_accuracy_series->attachAxis(m_axisY);
  m_accuracy_series->attachAxis(m_axisX);

  auto fileMenu = menuBar()->addMenu("File");
  openDataset = fileMenu->addAction("Open dataset...");

  connect(openDataset,
          &QAction::triggered,
          this,
          &QtMainWindowClass::handleOpenDataset);

  connect(this,
          &QtMainWindowClass::ClosingMainWindow,
          this,
          &QtMainWindowClass::stopTrainingOnQuit);

  connect(m_bttnTrain,
          &QPushButton::pressed,
          this,
          &QtMainWindowClass::startTraining);
}

void
QtMainWindowClass::stopTrainingOnQuit()
{
  m_quit_training = true;
}

void
QtMainWindowClass::closeEvent(QCloseEvent* event)
{
  emit ClosingMainWindow();
  event->accept();
}

void
QtMainWindowClass::train()
{
  size_t sampleSize = m_dataset.sample_size;
  size_t recordSize = m_dataset.input_size + 1;
  size_t inputSize = m_dataset.input_size;
  m_dataset.shuffle();

  for (size_t i = 0; i < sampleSize; i++) {
    for (size_t j = 0; j < inputSize; j++) {
      m_dataset.sample[i * inputSize + j] =
        m_dataset.record_values[i * recordSize + j];
      // sample[i + j] = atof(record.get(j + 1).c_str());
    }
    m_dataset.answers[i] =
      m_dataset.record_values[i * recordSize + recordSize - 1];
  }

  uint32_t n = 1;
  int i = 0;
  int update_counter = 0;
  int update_period = 100;
  m_accuracyList = QList<QPointF>(update_period);
  m_lossList = QList<QPointF>(update_period);
  while (i++ < 100 && !m_quit_training) {
    m_neuralNetwork->train();
    float accuracy = m_neuralNetwork->getAccuracy();
    float loss = m_neuralNetwork->getLoss();
    m_accuracyList[update_counter] = QPointF(n, accuracy);
    m_lossList[update_counter] = QPointF(n, loss);
    n++;
    m_datasetLabel->setText(std::to_string(i).c_str());
    if (update_counter++ >= update_period - 1) {
      update_counter = 0;
      m_loss_series->append(m_lossList);
      m_accuracy_series->append(m_accuracyList);
      /* QMetaObject::invokeMethod(
         this, &QtMainWindowClass::updateChart);*/
    }
    // QThread::sleep(10);
  }
}

void
QtMainWindowClass::handleOpenDataset()
{

  auto n =
    u"C:/Users/Ян/source/repos/Neu/examples/fisher_iris/samples/Iris.csv";
  auto fileName = QString::fromUtf16(
    u"C:/Users/Ян/source/repos/Neu/examples/fisher_iris/samples/Iris.csv");
  // auto fileName = QFileDialog::getOpenFileName(
  //   this, tr("Open Image"), "/", tr("Text files (*.csv)"));
  m_dataset_file.setFileName(fileName);
  if (!m_dataset_file.open(QIODeviceBase::ReadOnly | QIODeviceBase::Text)) {
    return;
  }
  m_loss_series->clear();
  m_accuracy_series->clear();
  m_datasetLabel->setText("Dataset file: " + fileName.split('/').last());

  CsvReader reader(fileName.toStdWString());
  auto record = reader.next();
  reader.setIdentifiers(record.value().get());
  size_t sampleSize = reader.getLineCount() - 1;
  size_t inputSize = 4;
  size_t recordSize = inputSize + 1;
  m_dataset = DataSet<5>(inputSize, sampleSize);
  auto& record_values = m_dataset.record_values;

  for (size_t i = 0; i < sampleSize; i++) {
    CsvReader::Record record = reader.next().value();
    for (size_t j = 0; j < recordSize - 1; j++) {
      record_values[i * recordSize + j] = atof(record.get(j + 1).c_str());
      // sample[i + j] = atof(record.get(j + 1).c_str());
    }
    if (record.get("Species").compare("Iris-setosa") == 0) {
      record_values[i * recordSize + recordSize - 1] = 0;
    } else if (record.get("Species").compare("Iris-versicolor") == 0) {
      record_values[i * recordSize + recordSize - 1] = 1;
    } else if (record.get("Species").compare("Iris-virginica") == 0) {
      record_values[i * recordSize + recordSize - 1] = 2;
    }
  }
}

void
QtMainWindowClass::clearChart()
{
  m_loss_series->clear();
  m_accuracy_series->clear();
}

void
QtMainWindowClass::startTraining()
{
  m_bttnTrain->setEnabled(false);

  m_datasetLabel->setText("1");
  clearChart();
  m_datasetLabel->setText(m_datasetLabel->text() + "1");

  if (m_boxLearningRate->value() == 0.0) {
    throw;
  }
  if (m_dataset.empty()) {
    throw;
  }
  m_datasetLabel->setText(m_datasetLabel->text() + "1");
  constexpr static std::initializer_list<size_t> configuration = { 4, 5, 5, 3 };

  m_neuralNetwork = std::make_unique<NeuralNetwork>(configuration);
  m_neuralNetwork->setLogger(nullptr)
    .setInputSample(m_dataset.sample.data(), m_dataset.sample_size)
    .setBatchSize(1)
    .setEpochs(1)
    .setLearningRate(m_boxLearningRate->value())
    //.setType(NeuralNetwork::NetworkType::CLASSIFIER)
    .setAnswers(m_dataset.answers.data())
    .setActivator(ParametricActivator<>::Sigmoid)
    .setLossFunction(LossFunction::CrossEntropy);
  m_datasetLabel->setText(m_datasetLabel->text() + "1");
  QtConcurrent::run(&QtMainWindowClass::train, this);
  m_bttnTrain->setEnabled(true);
}

QtMainWindowClass::~QtMainWindowClass() {}
