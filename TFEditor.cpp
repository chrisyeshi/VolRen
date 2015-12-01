/*
 * This file is part of HPGVis which is released under MIT.
 * See file LICENSE for full license details.
 */

#include "TFEditor.h"
#include <iostream>
#include <QSlider>
#include <QGridLayout>
#include <QColorDialog>

#ifndef nullptr
#define nullptr 0
#endif

namespace mslib {

namespace {

template <class T>
inline T clamp(T val, T minVal, T maxVal)
{
    return (val > maxVal ? maxVal : (val < minVal ? minVal : val));
}

inline float lerp(float a, float b, float w)
{
    return (a + w * (b - a));
}

inline int valueToIndex(double value, int resolution)
{
    return clamp((int)(value * (double)resolution), 0, resolution - 1);
}

inline double indexToValue(int index, int resolution)
{
    return (((double)index + 0.5) / (double)resolution);
}

inline QColor fromRgb(float r, float g, float b)
{
    QColor ret;
    ret.setRgbF(r, g, b);
    return ret;
}

inline QColor fromRgb(const float *rgb)
{
    QColor ret;
    ret.setRgbF(rgb[0], rgb[1], rgb[2]);
    return ret;
}

inline QColor fromRgba(const float *rgba)
{
    QColor ret;
    ret.setRgbF(rgba[0], rgba[1], rgba[2]);
    ret.setAlphaF(rgba[3]);
    return ret;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
//
//
//
//  Histogram
//
//
//
////////////////////////////////////////////////////////////////////////////////


Histogram::Histogram(unsigned int binCount)
    : _bins(nullptr),
      _binCount(binCount),
      _minVal(0.0),
      _maxVal(1.0)
{
    if (_binCount > 0)
        _bins = new unsigned int[_binCount]();
}

Histogram::~Histogram()
{
    if (_bins != nullptr) delete [] _bins;
}

Histogram &Histogram::operator = (const Histogram &other)
{
    if (_bins != nullptr) delete [] _bins;
    _binCount = other._binCount;
    _minVal = other._minVal;
    _maxVal = other._maxVal;
    _bins = new unsigned int[_binCount];
    for (unsigned int i = 0; i < _binCount; i++)
        _bins[i] = other._bins[i];
    return *this;
}

void Histogram::clear()
{
    for (unsigned int i = 0; i < _binCount; i++)
        _bins[i] = 0;
}

void Histogram::increment(double value)
{
    if (value < _minVal || value > _maxVal)
        return;
    double invRange = 1.0 / (_maxVal - _minVal);
    _bins[clamp((int)((value - _minVal) * invRange * (double)_binCount), 0, (int)_binCount - 1)]++;
}

////////////////////////////////////////////////////////////////////////////////
//
//
//
//  TFEditor
//
//
//
////////////////////////////////////////////////////////////////////////////////

TFEditor::TFEditor(QWidget *parent, int resolution, int arraySize)
    : QWidget(parent)
{
    _drawAreaLeftMargin = 8;
    _drawAreaRightMargin = 8;
    _colorMapAreaDrawAreaVerticalSpacing = 4;
    _drawAreaColorControlAreaVerticalSpacing = 4;

    bool enableVSlider = false;
    bool enableButtons = false;

    _tf = new TF(resolution, arraySize);
    _histogram = new Histogram(256);

    _colorMapArea = new TFColorMapArea(this);
    
    QSlider *_vSlider = new QSlider(Qt::Vertical, this);
    _vSlider->setRange(1, 9);
    _vSlider->setValue(5);
    _vSlider->setMaximumWidth(_drawAreaRightMargin);

    if (!enableVSlider)
    {
        _vSlider->hide();
    }
    
    _optionsMenu = new QMenu(tr("..."), this);
    QAction *openAction = _optionsMenu->addAction(tr("&Open..."));
    QAction *saveAsAction = _optionsMenu->addAction(tr("&Save As..."));

    _histogramMenu = _optionsMenu->addMenu(tr("Histogram"));
    _histogramShowAction = _histogramMenu->addAction(tr("Show"));
    _histogramShowAction->setCheckable(true);
    _histogramShowAction->setChecked(true);

    _bgColorMenu = _optionsMenu->addMenu(tr("Background Color"));
    QAction *bgBlackAction = _bgColorMenu->addAction(tr("&Black"));
    QAction *bgWhiteAction = _bgColorMenu->addAction(tr("&White"));

    _colorMapPresetsMenu = _optionsMenu->addMenu(tr("Color Map Presets"));
    QAction *rainbowMapAction = _colorMapPresetsMenu->addAction(tr("Rainbow Map"));

    _optionsButton = new QPushButton(tr("..."), this);
    _optionsButton->setFixedWidth(40);
    _optionsButton->setMenu(_optionsMenu);
    _gaussianButton = new QPushButton(tr("G"), this);
    _gaussianButton->setFixedWidth(30);

    if (!enableButtons)
    {
        _optionsButton->hide();
        _gaussianButton->hide();
    }
    
    _drawArea = new TFDrawArea(this);

    connect(_vSlider, SIGNAL(valueChanged(int)), _drawArea, SLOT(sliderValueChanged(int)));

    connect(_histogramShowAction, SIGNAL(changed()), _drawArea, SLOT(update()));
    connect(openAction, SIGNAL(triggered()), this, SLOT(open()));
    connect(saveAsAction, SIGNAL(triggered()), this, SLOT(saveAs()));
    connect(bgBlackAction, SIGNAL(triggered()), this, SLOT(setBGColorBlack()));
    connect(bgWhiteAction, SIGNAL(triggered()), this, SLOT(setBGColorWhite()));

    connect(rainbowMapAction, SIGNAL(triggered()), this, SLOT(applyPresetColorMap()));

    connect(_gaussianButton, SIGNAL(clicked()), this, SLOT(addGaussianObject()));

    _colorControlArea = new TFColorControlArea(this);
    _colorControlArea->setLeftMargin(_drawAreaLeftMargin);
    _colorControlArea->setRightMargin(_drawAreaRightMargin);

    QGridLayout *_layout = new QGridLayout();
    _layout->setSpacing(0);
    _layout->setColumnStretch(1, 1);
    _layout->addItem(new QSpacerItem(_drawAreaLeftMargin, 0, QSizePolicy::Fixed, QSizePolicy::Minimum), 0, 0);
    _layout->addWidget(_colorMapArea, 0, 1);
    _layout->addItem(new QSpacerItem(_drawAreaRightMargin, 0, QSizePolicy::Fixed, QSizePolicy::Minimum), 0, 2);
    _layout->addItem(new QSpacerItem(0, _colorMapAreaDrawAreaVerticalSpacing, QSizePolicy::Minimum, QSizePolicy::Fixed), 1, 0);
    _layout->addWidget(_drawArea, 2, 1);
    if (enableVSlider)
        _layout->addWidget(_vSlider, 2, 2);
    _layout->addItem(new QSpacerItem(0, _drawAreaColorControlAreaVerticalSpacing, QSizePolicy::Minimum, QSizePolicy::Fixed), 3, 0);
    _layout->addWidget(_colorControlArea, 4, 0, 1, 3);
    _layout->addItem(new QSpacerItem(0, _drawAreaColorControlAreaVerticalSpacing, QSizePolicy::Minimum, QSizePolicy::Fixed), 5, 0);
    if (enableButtons)
    {
        QHBoxLayout *bottomLayout = new QHBoxLayout();
        bottomLayout->addWidget(_optionsButton);
        bottomLayout->addWidget(_gaussianButton);
        bottomLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
        _layout->addLayout(bottomLayout, 6, 1);
    }

    _layout->setContentsMargins(0, 8, 0, 8);
    setLayout(_layout);
}

TFEditor::~TFEditor()
{
    delete _tf;
    delete _histogram;
}

void TFEditor::setTF(const TF &tf)
{
    *_tf = tf;
    updateTF(true, true);
    emit tfChanged();
    emit tfChanged(*_tf);
}

void TFEditor::setResolution(int resolution)
{
    if (_tf)
    {
        delete _tf;
        _tf = NULL;
    }
    _tf = new TF(resolution, resolution);
    updateTF(true, true);
    emitTFChanged();
}

void TFEditor::setHistogram(const Histogram &histogram)
{
    *_histogram = histogram;
    repaint();
}

void TFEditor::clearHistogram()
{
    _histogram->clear();
}

void TFEditor::loadTF(const QString &fileName)
{
    _tf->read(fileName.toLatin1().constData());
    updateTF(true, true);
    emitTFChanged();
}

void TFEditor::saveTF(const QString &fileName) const
{
    _tf->write(fileName.toLatin1().constData());
}

void TFEditor::updateTF(bool /*drawArray*/, bool colorControl)
{
    _tf->updateColorMap();
    _colorMapArea->updateImage();
    if (colorControl)
    {
        _drawArea->updateImage();
    }
    repaint();
}

void TFEditor::open()
{
    QString filePath = "";

    if (filePath != "")
        loadTF(filePath);
}

void TFEditor::saveAs()
{
    QString filePath = "";
    
    if (filePath != "")
        saveTF(filePath);
}

void TFEditor::addGaussianObject()
{
    float hf = 0.5f * 0.1f * sqrt(2.0f * 3.14159265358979323846f);
    _tf->addGaussianObject(0.5f, 0.1f, hf);
    _colorMapArea->updateImage();
    repaint();
    emitTFChanged();
}

void TFEditor::setBGColor(const QColor &color)
{
    const float *bgColor = _tf->backgroundColor();
    if (bgColor[0] != color.redF() ||
        bgColor[1] != color.greenF() ||
        bgColor[2] != color.blueF())
    {
        _tf->setBackgroundColor(color.redF(), color.greenF(), color.blueF());
        _colorMapArea->updateImage();
        repaint();
        emit bgColorChanged(color);
        emitTFChanged();
    }
}

void TFEditor::enableDrawArea(bool checked)
{
    _drawArea->setEnabled(checked);
    repaint();
}

void TFEditor::applyPresetColorMap()
{
    _tf->clearColorControls();
    _tf->addColorControl(TFColorControl(0.0f / 6.0f, 0.0f,      0.364706f, 1.0f));
    _tf->addColorControl(TFColorControl(1.0f / 6.0f, 0.0f,      1.0f,      0.976471f));
    _tf->addColorControl(TFColorControl(2.0f / 6.0f, 0.0f,      1.0f,      0.105882f));
    _tf->addColorControl(TFColorControl(3.0f / 6.0f, 0.968627f, 1.0f,      0.0f));
    _tf->addColorControl(TFColorControl(4.0f / 6.0f, 1.0f,      0.490196f, 0.0f));
    _tf->addColorControl(TFColorControl(5.0f / 6.0f, 1.0f,      0.0f,      0.0f));
    _tf->addColorControl(TFColorControl(6.0f / 6.0f, 0.662745f, 0.0f,      1.0f));
    _tf->updateColorMap();

    updateTF(false, true);
    emitTFChanged();
}

////////////////////////////////////////////////////////////////////////////////
//
//
//
//  TFColorMapArea
//
//
//
////////////////////////////////////////////////////////////////////////////////

TFColorMapArea::TFColorMapArea(TFEditor *parent)
    : QWidget(parent),
      _tfEditor(parent),
      _imageBuffer(nullptr)
{
    updateImage();
    setMinimumHeight(16);
    setMaximumHeight(16);
}

TFColorMapArea::~TFColorMapArea()
{
    if (_imageBuffer != nullptr) delete [] _imageBuffer;
}

void TFColorMapArea::mouseReleaseEvent(QMouseEvent *e)
{
    if (e->button() == Qt::RightButton)
        _tfEditor->execBGColorMenu(mapToGlobal(e->pos()));
}

void TFColorMapArea::paintEvent(QPaintEvent*)
{
    QPainter painter(this);

//    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    // set origin at left-bottom corner
    painter.translate(0.0, (double)height());
    painter.scale((double)width(), -(double)height());

    painter.setPen(Qt::NoPen);

    painter.drawImage(QRectF(0.0, 0.0, 1.0, 1.0), _image);
}

void TFColorMapArea::updateImage()
{
    if (_imageBuffer == nullptr)                                    // not initialized
        _imageBuffer = new unsigned char [tf().resolution() * 4];
    else if (_image.width() != tf().resolution())                   // size changed
    {
        delete [] _imageBuffer;
        _imageBuffer = new unsigned char [tf().resolution() * 4];
    }
    QColor bgColor = fromRgb(tf().backgroundColor());
    for (int i = 0; i < tf().resolution(); i++)
    {
        QColor color = fromRgba(&tf().colorMap()[i * 4]);
        QColor blendedColor;
        blendedColor.setRedF  (color.redF()   * color.alphaF() + bgColor.redF()   * (1.0 - color.alphaF()));
        blendedColor.setGreenF(color.greenF() * color.alphaF() + bgColor.greenF() * (1.0 - color.alphaF()));
        blendedColor.setBlueF (color.blueF()  * color.alphaF() + bgColor.blueF()  * (1.0 - color.alphaF()));
        // bgr1
        _imageBuffer[i * 4]     = blendedColor.blue();
        _imageBuffer[i * 4 + 1] = blendedColor.green();
        _imageBuffer[i * 4 + 2] = blendedColor.red();
        _imageBuffer[i * 4 + 3] = 255;
    }
    _image = QImage(_imageBuffer, tf().resolution(), 1, QImage::Format_RGB32);
}

////////////////////////////////////////////////////////////////////////////////
//
//
//
//  TFDrawArea
//
//
//
////////////////////////////////////////////////////////////////////////////////

TFDrawArea::TFDrawArea(TFEditor *parent)
    : QWidget(parent),
      _tfEditor(parent),
      _imageBuffer(nullptr),
      _draggedControl(),
      _changed(false)
{
    _showGrid = true;
    _vSliderValue = 5;
    _vSliderRange = 10;

    _gaussianControlRadius = 4.5;

    updateImage();

    setMouseTracking(true);
}

TFDrawArea::~TFDrawArea()
{
    if (_imageBuffer != nullptr) delete [] _imageBuffer;
}

void TFDrawArea::mouseMoveEvent(QMouseEvent *e)
{
    QPointF pos = getTransform().map(e->localPos());
    TFControlPoint ctrl = findControl(pos);
    setMouseCursor(ctrl);

    if (e->buttons() & Qt::LeftButton || e->buttons() & Qt::RightButton)
    {
        int lastIndex = valueToIndex(_lastPos.x(), tf().arraySize());
        float lastAlpha = tf().alphaArray()[lastIndex];
        int index = valueToIndex(pos.x(), tf().arraySize());

        if (e->buttons() & Qt::LeftButton)
        {
            setMouseCursor(_draggedControl);
            if (_draggedControl.type == TFControlPoint::GAUSSIAN_POSITION)
            {
                tf().gaussianObject(_draggedControl.index).mean = clamp(pos.x(), 0.0, 1.0);
                tf().gaussianObject(_draggedControl.index).update();
                _tfEditor->updateTF(true, false);
                _changed = true;
            }
            else if (_draggedControl.type == TFControlPoint::GAUSSIAN_HEIGHT)
            {
                tf().gaussianObject(_draggedControl.index).setHeight(clamp(ytoa((float)pos.y()), 0.0f, 100.0f));
                tf().gaussianObject(_draggedControl.index).update();
                _tfEditor->updateTF(true, false);
                _changed = true;
            }
            else if (_draggedControl.type == TFControlPoint::GAUSSIAN_WIDTH_LEFT)
            {
                float mean = tf().gaussianObject(_draggedControl.index).mean;
                float ht = tf().gaussianObject(_draggedControl.index).height();
                tf().gaussianObject(_draggedControl.index).sigma = clamp(mean - (float)pos.x(), 0.0001f, 100.0f);
                tf().gaussianObject(_draggedControl.index).setHeight(ht);
                tf().gaussianObject(_draggedControl.index).update();
                _tfEditor->updateTF(true, false);
                _changed = true;
            }
            else if (_draggedControl.type == TFControlPoint::GAUSSIAN_WIDTH_RIGHT)
            {
                float mean = tf().gaussianObject(_draggedControl.index).mean;
                float ht = tf().gaussianObject(_draggedControl.index).height();
                tf().gaussianObject(_draggedControl.index).sigma = clamp((float)pos.x() - mean, 0.0001f, 100.0f);
                tf().gaussianObject(_draggedControl.index).setHeight(ht);
                tf().gaussianObject(_draggedControl.index).update();
                _tfEditor->updateTF(true, false);
                _changed = true;
            }
            else
            {
                float alpha = ytoa(clamp((float)pos.y(), 0.0f, 1.0f));
                int step = (index <= lastIndex ? 1 : -1);
                for (int i = index; i != lastIndex; i += step)
                {
                    tf().alphaArray()[i] = lerp(alpha, lastAlpha, (float)std::abs(i - index) / (float)std::abs(lastIndex - index));
                }
                tf().alphaArray()[index] = alpha;       // for the case index = lastIndex
                _tfEditor->updateTF(true, false);
                _changed = true;
            }
            _lastPos = pos;
            e->accept();
        }
        else if (e->buttons() & Qt::RightButton)
        {
            int step = (index <= lastIndex ? 1 : -1);
            for (int i = index; i != lastIndex; i += step)
            {
                tf().alphaArray()[i] = lerp(0.0f, lastAlpha, (float)std::abs(i - index) / (float)std::abs(lastIndex - index));
            }
            tf().alphaArray()[index] = 0.0f;            // for the case index = lastIndex
            _lastPos = pos;
            _tfEditor->updateTF(true, false);
            _changed = true;
            e->accept();
        }
    }

//    if (_changed)
//    {
//        _changed = false;
//        _tfEditor->emitTFChanged();
//    }
}

void TFDrawArea::mousePressEvent(QMouseEvent *e)
{
    QPointF pos = getTransform().map(e->localPos());
    int index = valueToIndex(pos.x(), tf().arraySize());
    if (e->buttons() & Qt::LeftButton)
    {
        TFControlPoint ctrl = findControl(pos);
        if (ctrl.type == TFControlPoint::GAUSSIAN_POSITION ||
            ctrl.type == TFControlPoint::GAUSSIAN_HEIGHT ||
            ctrl.type == TFControlPoint::GAUSSIAN_WIDTH_LEFT ||
            ctrl.type == TFControlPoint::GAUSSIAN_WIDTH_RIGHT)
        {
            _draggedControl = ctrl;
        }
        else
        {
            float alpha = ytoa(clamp((float)pos.y(), 0.0f, 1.0f));
            tf().alphaArray()[index] = alpha;
            _tfEditor->updateTF(true, false);
            _changed = true;
            
        }
        _lastPos = pos;
        e->accept();
    }
    else if (e->buttons() & Qt::RightButton)
    {
        TFControlPoint ctrl = findControl(pos);
        if (ctrl.type == TFControlPoint::GAUSSIAN_POSITION)
        {
            tf().removeGaussianObject(ctrl.index);
            _tfEditor->updateTF(true, false);
            _changed = true;
        }
        else
        {
            tf().alphaArray()[index] = 0.0f;
            _tfEditor->updateTF(true, false);
            _changed = true;
        }
        _lastPos = pos;
        e->accept();
    }
}

void TFDrawArea::mouseReleaseEvent(QMouseEvent *e)
{
    _draggedControl.type = TFControlPoint::VOID_CONTROL;

    QPointF pos = getTransform().map(e->localPos());
    TFControlPoint ctrl = findControl(pos);
    setMouseCursor(ctrl);

    if (_changed)
    {
        _changed = false;
        _tfEditor->emitTFChanged();
    }
}

void TFDrawArea::paintEvent(QPaintEvent*)
{
    QPainter painter(this);

    painter.setRenderHint(QPainter::Antialiasing, true);

    // set origin at left-bottom corner
    painter.save();
    painter.translate(0.0, (double)height());
    painter.scale((double)width(), -(double)height());

    // color map image
    painter.setPen(Qt::NoPen);

    painter.drawImage(QRectF(0.0, 0.0, 1.0, 1.0), _image);

    // grid line
    if (_showGrid)
    {
        QPen gridPen(Qt::DashLine);
        gridPen.setWidthF(0.f);
        gridPen.setColor(Qt::white);

        painter.setPen(gridPen);
        painter.setBrush(QBrush(QColor(255, 255, 255), Qt::SolidPattern));
        float y = atoy(0.5f);
        painter.drawLine(QPointF(0.0, y), QPointF(1.0, y));

        gridPen.setColor(Qt::gray);
        painter.setPen(gridPen);
        for (int i = 1; i <= 9; i++)
        {
            if (i == 5)
                continue;
            float y = atoy(0.1f * (float)i);
            painter.drawLine(QPointF(0.0, y), QPointF(1.0, y));
        }
    }

    // histogram
    if (_tfEditor->histogramEnabled() && _tfEditor->getHistogram().binCount() > 0)
    {
        painter.setPen(Qt::NoPen);
        painter.setBrush(QBrush(QColor(128, 128, 128, 128), Qt::SolidPattern));

        Histogram &hist = _tfEditor->getHistogram();
        unsigned int maxBin = 0;
        for (unsigned int i = 0; i < hist.binCount(); i++)
            if (maxBin < hist[i])
                maxBin = hist[i];

        double rectW = 1.0 / (double)hist.binCount();
        for (unsigned int i = 0; i < hist.binCount(); i++)
        {
            double x = (double)i / (double)hist.binCount();
            double rectH = (maxBin == 0) ? 0.0 : (double)hist[i] / (double)maxBin;
            painter.drawRect(QRectF(x, 0.0, rectW, rectH));
        }
    }

    int res = tf().resolution();
    QPointF *points = new QPointF[res + 4];
    points[0] = QPointF(0.0, 0.0);
    points[res + 3] = QPointF(1.0, 0.0);

    // gaussian
    for (int i = 0; i < tf().gaussianObjectCount(); i++)
    {
        TFGaussianObject &gaus = tf().gaussianObject(i);
        points[1] = QPointF(0.0, atoy(gaus.alphaArray[0]));
        for (int i = 0; i < res; i++)
            points[i + 2] = QPointF(indexToValue(i, res), atoy(gaus.alphaArray[i]));
        points[res + 2] = QPointF(1.0, atoy(gaus.alphaArray[res - 1]));

        // semi-transparent area
        painter.setPen(Qt::NoPen);
        painter.setBrush(QBrush(QColor(255, 255, 255, 96), Qt::SolidPattern));
        painter.drawPolygon(points, res + 4);

        // the curve
        QPen pen;
        pen.setWidth(0);
        pen.setColor(Qt::black);
        painter.setPen(pen);
        painter.setBrush(QBrush(QColor(255, 255, 255), Qt::SolidPattern));
        painter.drawPolyline(&points[1], res + 2);
    }

    int size = tf().arraySize();
    delete [] points;
    points = new QPointF[size * 2 + 2];
    points[0] = QPointF(0.0, 0.0);

    // hand-drawn step curve
    for (int i = 0; i < size; ++i)
    {
        points[2 * i + 1] = QPointF(double(i+0) / double(size), atoy(tf().alphaArray()[i]));
        points[2 * i + 2] = QPointF(double(i+1) / double(size), atoy(tf().alphaArray()[i]));
    }
    points[2 * size + 1] = QPointF(1.0, 0.0);
    
    // semi-transparent area
    painter.setPen(Qt::NoPen);
    painter.setBrush(QBrush(QColor(255, 255, 255, 96), Qt::SolidPattern));
    painter.drawPolygon(points, size * 2 + 2);

    QPen pen;
    pen.setWidth(0);
    // the curve
    pen.setColor(Qt::black);
    painter.setPen(pen);
    painter.setBrush(QBrush(QColor(255, 255, 255), Qt::SolidPattern));
    painter.drawPolyline(&points[1], size * 2);

    delete [] points;

    // gaussian control points
    painter.setPen(QPen(Qt::black));
    painter.setBrush(QBrush(QColor(255, 255, 255), Qt::SolidPattern));
    for (int i = 0; i < tf().gaussianObjectCount(); i++)
    {
        // TFGaussianObject &gaus = tf().gaussianObject(i);

        TFControlPoint posCtrl = getGaussianPositionControl(i);

        QPointF r = QPointF(_gaussianControlRadius / (double)width(), _gaussianControlRadius / (double)height());
        painter.drawRect(QRectF(posCtrl.pos.x() - r.x(), posCtrl.pos.y() - r.y(), r.x() * 2.0, r.y() * 2.0));
        
        // height control
        TFControlPoint hCtrl = getGaussianHeightControl(i);
        painter.drawEllipse(QPointF(hCtrl.pos.x(), clamp(atoy(hCtrl.pos.y()), 0.0f, 1.0f)), r.x(), r.y());

        // width controls
        TFControlPoint wlCtrl = getGaussianWidthLeftControl(i);
        TFControlPoint wrCtrl = getGaussianWidthRightControl(i);
        painter.drawEllipse(QPointF(clamp(wlCtrl.pos.x(), 0.0, 1.0), wlCtrl.pos.y()), r.x(), r.y());
        painter.drawEllipse(QPointF(clamp(wrCtrl.pos.x(), 0.0, 1.0), wrCtrl.pos.y()), r.x(), r.y());
    }

    // grey out
    if (!this->isEnabled())
    {
        painter.setPen(Qt::NoPen);
        painter.setBrush(QBrush(QColor(25, 25, 25, 150), Qt::SolidPattern));
        painter.drawRect(QRectF(0.0, 0.0, 1.0, 1.0));
    }

    painter.restore();
}

QTransform TFDrawArea::getTransform() const
{
    QTransform tm;
    tm.translate(0.0, (double)height());
    tm.scale((double)width(), -(double)height());
    return tm.inverted();
}

void TFDrawArea::setMouseCursor(const TFControlPoint &ctrl)
{
    switch (ctrl.type)
    {
    case TFControlPoint::GAUSSIAN_POSITION:    setCursor(Qt::SizeAllCursor); break;
    case TFControlPoint::GAUSSIAN_HEIGHT:      setCursor(Qt::SizeVerCursor); break;
    case TFControlPoint::GAUSSIAN_WIDTH_LEFT:
    case TFControlPoint::GAUSSIAN_WIDTH_RIGHT: setCursor(Qt::SizeHorCursor); break;
    default:                                   setCursor(Qt::ArrowCursor);   break;
    }
}

TFControlPoint TFDrawArea::getGaussianPositionControl(int index) const
{
    return TFControlPoint(TFControlPoint::GAUSSIAN_POSITION, index,
                          QPointF(tf().gaussianObject(index).mean, _gaussianControlRadius / height()));
}

TFControlPoint TFDrawArea::getGaussianHeightControl(int index) const
{
    return TFControlPoint(TFControlPoint::GAUSSIAN_HEIGHT, index,
                          QPointF(tf().gaussianObject(index).mean, tf().gaussianObject(index).height()));
}

TFControlPoint TFDrawArea::getGaussianWidthLeftControl(int index) const
{
    return TFControlPoint(TFControlPoint::GAUSSIAN_WIDTH_LEFT, index,
                          QPointF(tf().gaussianObject(index).mean - tf().gaussianObject(index).sigma, _gaussianControlRadius / height()));
}

TFControlPoint TFDrawArea::getGaussianWidthRightControl(int index) const
{
    return TFControlPoint(TFControlPoint::GAUSSIAN_WIDTH_RIGHT, index,
                          QPointF(tf().gaussianObject(index).mean + tf().gaussianObject(index).sigma, _gaussianControlRadius / height()));
}

TFControlPoint TFDrawArea::findControl(const QPointF &pos) const
{
    TFControlPoint ret;

    double controlSizeX = _gaussianControlRadius / width();
    double controlSizeY = _gaussianControlRadius / height();

    for (int i = tf().gaussianObjectCount() - 1; i >= 0; i--)       // find last
    {
        TFControlPoint hCtrl = getGaussianHeightControl(i);
        if (QRectF(hCtrl.pos.x() - controlSizeX, clamp(atoy(hCtrl.pos.y()), 0.0f, 1.0f) - controlSizeY,
                   controlSizeX * 2.0, controlSizeY * 2.0).contains(pos))
        {
            ret = hCtrl;
            break;
        }

        TFControlPoint wlCtrl = getGaussianWidthLeftControl(i);
        if (QRectF(clamp(wlCtrl.pos.x(), 0.0, 1.0) - controlSizeX, wlCtrl.pos.y() - controlSizeY,
                   controlSizeX * 2.0, controlSizeY * 2.0).contains(pos))
        {
            ret = wlCtrl;
            break;
        }

        TFControlPoint wrCtrl = getGaussianWidthRightControl(i);
        if (QRectF(clamp(wrCtrl.pos.x(), 0.0, 1.0) - controlSizeX, wrCtrl.pos.y() - controlSizeY,
                   controlSizeX * 2.0, controlSizeY * 2.0).contains(pos))
        {
            ret = wrCtrl;
            break;
        }

        TFControlPoint posCtrl = getGaussianPositionControl(i);
        if (QRectF(posCtrl.pos.x() - controlSizeX, posCtrl.pos.y() - controlSizeY,
                   controlSizeX * 2.0, controlSizeY * 2.0).contains(pos))
        {
            ret = posCtrl;
            break;
        }
    }
    return ret;
}

float TFDrawArea::atoy(float alpha) const
{
    float stepSize = 1.0f / (float)_vSliderRange;       // 0.1
    float halfRange = (float)(_vSliderRange / 2);       // 5
    if (alpha <= stepSize * (float)_vSliderValue)
        return alpha / (float)_vSliderValue * halfRange;
    else
        return (alpha - stepSize * (float)_vSliderValue) / (float)(_vSliderRange - _vSliderValue) * halfRange + 0.5f;
}

float TFDrawArea::ytoa(float y) const
{
    float stepSize = 1.0f / (float)_vSliderRange;       // 0.1
    float halfRange = (float)(_vSliderRange / 2);       // 5
    if (y <= 0.5f)
        return y / halfRange * (float)_vSliderValue;
    else
        return (y - 0.5f) / halfRange * (float)(_vSliderRange - _vSliderValue) + stepSize * (float)_vSliderValue;
}

void TFDrawArea::updateImage()
{
    if (_imageBuffer == nullptr)                                    // not initialized
        _imageBuffer = new unsigned char [tf().resolution() * 4];
    else if (_image.width() != tf().resolution())                   // size changed
    {
        delete [] _imageBuffer;
        _imageBuffer = new unsigned char [tf().resolution() * 4];
    }
    for (int i = 0; i < tf().resolution(); i++)
    {
        QColor color = fromRgb(&tf().colorMap()[i * 4]);
        // bgr1
        _imageBuffer[i * 4]     = color.blue();
        _imageBuffer[i * 4 + 1] = color.green();
        _imageBuffer[i * 4 + 2] = color.red();
        _imageBuffer[i * 4 + 3] = 255;
    }
    _image = QImage(_imageBuffer, tf().resolution(), 1, QImage::Format_RGB32);
}

void TFDrawArea::sliderValueChanged(int value)
{
    _vSliderValue = value;
    repaint();
}

////////////////////////////////////////////////////////////////////////////////
//
//
//
//  TFColorControlArea
//
//
//
////////////////////////////////////////////////////////////////////////////////

TFColorControlArea::TFColorControlArea(TFEditor *parent)
    : QWidget(parent),
      _tfEditor(parent)
{
    _leftMargin = 10;
    _rightMargin = 10;
    _controlWidth = 8;

    _activeControl = -1;
    _hoveredControl = -1;
    _draggedControl = -1;

    setMinimumHeight(16);
    setMaximumHeight(16);

    setMouseTracking(true);
}

void TFColorControlArea::leaveEvent(QEvent*)
{
    if (_hoveredControl >= 0)
    {
        _hoveredControl = -1;
        repaint();
    }
}

void TFColorControlArea::mouseDoubleClickEvent(QMouseEvent *e)
{
    if (e->buttons() & Qt::LeftButton)
    {
        QPointF pos = getTransform().map(e->localPos());
        int index = findControl(pos);
        if (index >= 0)
        {
            QColor color = fromRgb(tf().colorControl(index).color);
            QColor newColor = QColorDialog::getColor(color, this);
            if (newColor.isValid())
            {
                tf().colorControl(index).color[0] = (float)newColor.redF();
                tf().colorControl(index).color[1] = (float)newColor.greenF();
                tf().colorControl(index).color[2] = (float)newColor.blueF();
                _tfEditor->updateTF(false, true);
                _tfEditor->emitTFChanged();
            }
            e->accept();
        }
        else if (pos.x() >= 0.0 && pos.y() <= 1.0)
        {
            float value = (float)pos.x();
            tf().insertColorControl(value);
            _activeControl = tf().colorControlCount() - 1;
            _tfEditor->updateTF(false, true);
            _tfEditor->emitTFChanged();
            e->accept();
        }
    }
}

void TFColorControlArea::mouseMoveEvent(QMouseEvent *e)
{
    QTransform tm = getTransform();
    QPointF pos = tm.map(e->localPos());
    double controlWidthF = (double)_controlWidth / (double)(width() - _leftMargin - _rightMargin);

    if (e->buttons() & Qt::LeftButton)
    {
        if (_draggedControl >= 0)
        {
            float newValue = tf().colorControl(_draggedControl).value - _lastPos.x() + pos.x();
            newValue = clamp(newValue, 0.0f, 1.0f);
            tf().colorControl(_draggedControl).value = newValue;
            _lastPos = pos;
            _lastPos.setX(clamp(_lastPos.x(), -controlWidthF, 1.0 + controlWidthF));
            _tfEditor->updateTF(false, true);
            _tfEditor->emitTFChanged();
            e->accept();
        }
    }
    else
    {
        int index = findControl(pos);
        
        if (index >= 0)
        {
            _hoveredControl = index;
            repaint();
            e->accept();
        }
        else if (_hoveredControl >= 0)
        {
            _hoveredControl = -1;
            repaint();
            e->accept();
        }
    }
}

void TFColorControlArea::mousePressEvent(QMouseEvent *e)
{
    if (e->buttons() & Qt::LeftButton)
    {
        QPointF pos = getTransform().map(e->localPos());
        int index = findControl(pos);
        _activeControl = index;
        _draggedControl = index;
        _lastPos = pos;
        repaint();
        e->accept();
    }
    else if (e->buttons() & Qt::RightButton)
    {
        QPointF pos = getTransform().map(e->localPos());
        int index = findControl(pos);
        if (index >= 0 && tf().colorControlCount() > 1)
        {
            tf().removeColorControl(index);
            if (_activeControl == tf().colorControlCount())
                _activeControl--;
            _tfEditor->updateTF(false, true);
            _tfEditor->emitTFChanged();
        }
        e->accept();
    }
}

void TFColorControlArea::mouseReleaseEvent(QMouseEvent*)
{
    if (_draggedControl >= 0)
        _draggedControl = -1;
}

void TFColorControlArea::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.translate((double)_leftMargin, (double)height());
    painter.scale((double)(width() - _leftMargin - _rightMargin), -(double)height());

    painter.setPen(Qt::NoPen);
    QBrush brush(QColor(192, 192, 192), Qt::SolidPattern);
    painter.setBrush(brush);

    painter.drawRect(QRectF(0.0, 0.0, 1.0, 1.0));

    painter.setBrush(QBrush(QColor(255, 255, 255), Qt::SolidPattern));
    
    double controlWidthF = (double)_controlWidth / (double)(width() - _leftMargin - _rightMargin);

    for (int i = 0; i < tf().colorControlCount(); i++)
    {
        double pos = (double)tf().colorControl(i).value;
        QPointF points[] = {QPointF(pos - controlWidthF, 0.0),
                            QPointF(pos + controlWidthF, 0.0),
                            QPointF(pos, 1.0)};
        QColor color = fromRgb(tf().colorControl(i).color);
        painter.setBrush(QBrush(color, Qt::SolidPattern));
        QPen pen;
        pen.setWidthF(0.f);
        if (i == _activeControl || i == _hoveredControl)
            pen.setColor(Qt::white);
        else
            pen.setColor(Qt::black);
        painter.setPen(pen);
        painter.drawPolygon(points, 3);
    }

    if (_activeControl >= 0)
    {
        double pos = (double)tf().colorControl(_activeControl).value;
        QPointF points[] = {QPointF(pos - controlWidthF, 0.0),
                            QPointF(pos + controlWidthF, 0.0),
                            QPointF(pos, 1.0)};
        QColor color = fromRgb(tf().colorControl(_activeControl).color);
        painter.setBrush(QBrush(color, Qt::SolidPattern));
        QPen pen;
        pen.setWidthF(0.f);
        pen.setColor(Qt::white);
        painter.setPen(pen);
        painter.drawPolygon(points, 3);
    }
}

QTransform TFColorControlArea::getTransform() const
{
    QTransform tm;
    tm.translate((double)_leftMargin, (double)height());
    tm.scale((double)(width() - _leftMargin - _rightMargin), -(double)height());
    return tm.inverted();
}

int TFColorControlArea::findControl(const QPointF &pos) const
{
    float controlWidthF = (double)_controlWidth / (double)(width() - _leftMargin - _rightMargin);
    for (int i = tf().colorControlCount() - 1; i >= 0; i--)         // find last
    {
        float controlPos = tf().colorControl(i).value;
        if (pos.x() >= controlPos - controlWidthF &&
            pos.x() <= controlPos + controlWidthF &&
            pos.y() >= 0.0 &&
            pos.y() <= 1.0)
        {
            return i;
        }
    }
    return -1;
}

} // namespace mslib
