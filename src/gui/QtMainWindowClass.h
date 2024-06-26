#pragma once

#include "Neu/CsvReader.h"
#include "Neu/NeuralNetwork.h"
#include "ui_QtMainWindowClass.h"
#include <QChartView>
#include <QLineSeries>
#include <QMainWindow>
#include <QValueAxis>
#include <QtConcurrent/QtConcurrent>
#include <QtWidgets>
#include <Windows.h>
#include <string>

class QtMainWindowClass : public QMainWindow
{
  Q_OBJECT

public:
  QtMainWindowClass(QWidget* parent = nullptr);
  ~QtMainWindowClass();

private:
  template<size_t recordSize>
  struct DataSet
  {
    using record_ptr_t = std::byte (*)[recordSize * sizeof(float)];
    std::vector<float> record_values;
    std::vector<float> sample;
    std::vector<float> answers;
    size_t input_size;
    size_t sample_size;

    DataSet()
      : input_size(0)
      , sample_size(0)
    {
    }
    DataSet(size_t inputSize, size_t sampleSize)
      : record_values((inputSize + 1) * sampleSize)
      , answers(sampleSize)
      , sample(inputSize * sampleSize)
      , input_size(inputSize)
      , sample_size(sampleSize)
    {
    }

    void shuffle()
    {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(reinterpret_cast<record_ptr_t>(record_values.data()),
                   reinterpret_cast<record_ptr_t>(record_values.data()) +
                     sample_size,
                   g);
    }

    bool empty() { return sample.empty(); }
  };

signals:
  void ClosingMainWindow();

public slots:
  void stopTrainingOnQuit();

  void handleOpenDataset();

  void startTraining();

private:
  void closeEvent(QCloseEvent* event) override;

	void clearChart();

  void train();

  Ui::QtMainWindowClass ui;
  QFile m_dataset_file;
  QLabel* m_datasetLabel;
  QChartView* m_training_chart_view;
  QLineSeries* m_accuracy_series;
  QLineSeries* m_loss_series;
  bool m_quit_training = false;
  QValueAxis* m_axisY;
  QValueAxis* m_axisX;
  QHBoxLayout* horizontalLayout;
  QVBoxLayout* m_verticalLayout;

  QLabel* m_lbllearningRate;
  QDoubleSpinBox* m_boxLearningRate;
  QPushButton* m_bttnTrain;
  QWidget* m_verticalLayoutFill;

  QList<QPointF> m_accuracyList;
  QList<QPointF> m_lossList;
  DataSet<5> m_dataset;
  std::unique_ptr<NeuralNetwork> m_neuralNetwork;
};
