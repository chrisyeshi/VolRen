/*
 * This file is part of HPGVis which is released under MIT.
 * See file LICENSE for full license details.
 */

#ifndef TFEDITOR_H
#define TFEDITOR_H

#include <QtGui>
#include <QMenu>
#include <QPushButton>

#include "TF.h"

namespace mslib {

class TFColorMapArea;
class TFDrawArea;
class TFColorControlArea;

struct TFControlPoint
{
    enum Type
    {
        VOID_CONTROL,
        GAUSSIAN_POSITION,
        GAUSSIAN_HEIGHT,
        GAUSSIAN_WIDTH_LEFT,
        GAUSSIAN_WIDTH_RIGHT
    };
    TFControlPoint() : type(VOID_CONTROL), index(-1), pos() {}
    TFControlPoint(Type _type, int _index, const QPointF &_pos) : type(_type), index(_index), pos(_pos) {}
    Type type;
    int index;
    QPointF pos;
};

class Histogram
{
public:
    Histogram(unsigned int binCount = 256);
    Histogram(const Histogram &other) { *this = other; }
    ~Histogram();

    Histogram &operator = (const Histogram &other);
    const unsigned int &operator [] (unsigned int index) const { return _bins[index]; }

    void clear();

    unsigned int binCount() const { return _binCount; }
    double getMin() const { return _minVal; }
    double getMax() const { return _maxVal; }

    void setRange(double minVal, double maxVal) { _minVal = minVal; _maxVal = maxVal; }

    void increment(double value);
    
private:
    unsigned int *_bins;
    unsigned int _binCount;
    double _minVal, _maxVal;
};

////////////////////////////////////////////////////////////////////////////////

class TFEditor : public QWidget
{
    Q_OBJECT

    friend class TFColorMapArea;
    friend class TFDrawArea;
    friend class TFColorControlArea;
public:
    TFEditor(QWidget *parent = 0, int resolution = 1024);
    virtual ~TFEditor();

    std::shared_ptr<TF const> getTF() const { return _tf; }
    void setTF(std::shared_ptr<TF> tf);
    void setTF(const TF& tf);

    bool histogramEnabled() const { return _histogramShowAction->isChecked(); }
    Histogram       &getHistogram()       { return *_histogram; }
    const Histogram &getHistogram() const { return *_histogram; }
    void setHistogram(const Histogram &histogram);
    void clearHistogram();
    void incrementHistogram(double value) { _histogram->increment(value); }     // need to repaint

    // mouse tracking
    void setTracking(bool isTracking) { _isTracking = isTracking; }

    // I/O
    void loadTF(const QString &fileName);
    void saveTF(const QString &fileName) const;

public slots:
    void setResolution(int resolution);

protected:
    QAction *execBGColorMenu(const QPoint &p, QAction *action = 0) { return _bgColorMenu->exec(p, action); }

    void updateTF(bool drawArray = true, bool colorControl = true);

    void emitTFChanged(bool makeHistory = false) { emit tfChanged(); emit tfChanged(makeHistory); emit tfChanged(*_tf); }

private:
    std::shared_ptr<TF> _tf;
    Histogram *_histogram;

    TFColorMapArea *_colorMapArea;
    TFDrawArea *_drawArea;
    TFColorControlArea *_colorControlArea;
    
    QMenu       *_optionsMenu;
    QMenu       *_histogramMenu;
    QAction     *_histogramShowAction;
    QMenu       *_bgColorMenu;
    QMenu       *_colorMapPresetsMenu;
    QPushButton *_optionsButton;
    QPushButton *_gaussianButton;

    int _drawAreaLeftMargin;
    int _drawAreaRightMargin;
    int _colorMapAreaDrawAreaVerticalSpacing;
    int _drawAreaColorControlAreaVerticalSpacing;
    bool _isTracking;

protected slots:
    void open();
    void saveAs();
    void addGaussianObject();
    void setBGColorBlack() { setBGColor(QColor(0, 0, 0, 255)); }
    void setBGColorWhite() { setBGColor(QColor(255, 255, 255, 255)); }
    void applyPresetColorMap();

public slots:
    void setBGColor(const QColor &color);
    void enableDrawArea(bool checked);

signals:
    void tfChanged();
    void tfChanged(bool makeHistory);
    void tfChanged(const mslib::TF &tf);
    void bgColorChanged(const QColor &color);
};

////////////////////////////////////////////////////////////////////////////////

class TFColorMapArea : public QWidget
{
    Q_OBJECT

    friend class TFEditor;
public:
    TFColorMapArea(TFEditor *parent = 0);
    ~TFColorMapArea();

protected:
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void paintEvent(QPaintEvent *e);

    TF &tf() { return *(_tfEditor->_tf); }
    void updateImage();

private:
    TFEditor *_tfEditor;

    QImage _image;
    unsigned char *_imageBuffer;
};

////////////////////////////////////////////////////////////////////////////////

class TFDrawArea : public QWidget
{
    Q_OBJECT

    friend class TFEditor;
public:
    TFDrawArea(TFEditor *parent = 0);
    ~TFDrawArea();

protected:
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void mousePressEvent(QMouseEvent *e);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void paintEvent(QPaintEvent *e);

    TF& tf() { return *(_tfEditor->_tf); }
    const TF &tf() const { return *(_tfEditor->_tf); }
    QTransform getTransform() const;

    void setMouseCursor(const TFControlPoint &ctrl);

    TFControlPoint getGaussianPositionControl(int index) const;
    TFControlPoint getGaussianHeightControl(int index) const;
    TFControlPoint getGaussianWidthLeftControl(int index) const;
    TFControlPoint getGaussianWidthRightControl(int index) const;

    TFControlPoint findControl(const QPointF &pos) const;
        
    float atoy(float alpha) const;      // depends on vSliderValue
    float ytoa(float y) const;          // depends on vSliderValue

    void updateImage();

private:
    TFEditor *_tfEditor;

    QImage _image;
    unsigned char *_imageBuffer;
    QPointF _lastPos;       // mouse moving

    bool _showGrid;
    int _vSliderValue;
    int _vSliderRange;

    double _gaussianControlRadius;

    TFControlPoint _draggedControl;

    bool _changed;

protected slots:
    void sliderValueChanged(int value);
};

////////////////////////////////////////////////////////////////////////////////

class TFColorControlArea : public QWidget
{
    Q_OBJECT

    friend class TFEditor;
public:
    TFColorControlArea(TFEditor *parent = 0);
    void setLeftMargin(int leftMargin) { _leftMargin = leftMargin; }
    void setRightMargin(int rightMargin) { _rightMargin = rightMargin; }

protected:
    virtual void leaveEvent(QEvent *e);
    virtual void mouseDoubleClickEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void mousePressEvent(QMouseEvent *e);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void paintEvent(QPaintEvent *e);

    TF       &tf()       { return *(_tfEditor->_tf); }
    const TF &tf() const { return *(_tfEditor->_tf); }
    QTransform getTransform() const;        // window coordinates to painting coordinates
    int findControl(const QPointF &pos) const;

private:
    TFEditor *_tfEditor;

    int _leftMargin;
    int _rightMargin;
    int _controlWidth;      // half width of the triangle stuffs which control the color

    int _activeControl;
    int _hoveredControl;
    int _draggedControl;
    QPointF _lastPos;
};

} // namespace mslib

typedef mslib::TFEditor TFEditor;

#endif // TFEDITOR_H
