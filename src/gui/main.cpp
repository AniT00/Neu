#include <Qapplication>
#include <QtWidgets>
#include <QtCharts/qchart.h>
#include <QtCharts/qchartview.h>
#include <QtCharts/qlineseries.h>
#include <qstyle.h>
#include "QtMainWindowClass.h"
#include <Windows.h>

int
main(int argc, char** argv)
{
  QApplication app(argc, argv);
  auto window = new QtMainWindowClass();
  app.setStyle("Plastique");

  window->resize(720, 480);
  window->show();
  window->setWindowTitle(
    QApplication::translate("toplevel", "Top-level widget"));

  return app.exec();
}