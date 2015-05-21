#include "imagewriterraf.h"
#include <fstream>
#include <cassert>
#include <QtOpenGL>
#include <QOpenGLFunctions_3_3_Core>

namespace yy {
namespace volren {

void ImageWriterRAF::write(const std::string &name) const
{
    std::string filename = name;
    std::ofstream fout(filename, std::ios::binary);
    int format = 0x201001;
    int type = 0x1406;
    int realnum = 1;
    std::vector<float> alphas(image->layers);
    fout.write(reinterpret_cast<const char*>(&image->w), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&image->h), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&format), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&type), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&realnum), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&image->layers), sizeof(int));
    fout.write(reinterpret_cast<const char*>(alphas.data()), alphas.size() * sizeof(float));
    // get data from PBOs
    int nNums = image->w * image->h * image->layers;
    int nBytes = nNums * sizeof(float);
    std::vector<float> raf(nNums);
    std::vector<float> dep(nNums);
    QOpenGLFunctions_3_3_Core* f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
    f->initializeOpenGLFunctions();
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *image->rafPBO);
    f->glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, nBytes, raf.data());
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *image->depPBO);
    f->glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, nBytes, dep.data());
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // write
    fout.write(reinterpret_cast<const char*>(raf.data()), nBytes);
    fout.write(reinterpret_cast<const char*>(dep.data()), nBytes);
}

} // namespace volren
} // namespace yy