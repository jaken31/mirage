#include "gl_context.h"

#include <cstring>
#include <cstdio>
#include <GLFW/glfw3.h>

namespace {
    void GlfwErrorCallback(int error, const char* description) {
        std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
    }
    const char* const kSoftwareGl[] = {
        "GDI Generic",
        "Microsoft Basic Render Driver",
    };

}

GlContext::GlContext(const mjModel* model) {
    glfwSetErrorCallback(GlfwErrorCallback);

    if (!glfwInit()) {
        mju_error("Failed to initialize GLFW");
    }

    // The window is never drawn into - it exists only to own the GL context.
    // Single-buffered because nothing is ever swapped.
    glfwWindowHint(GLFW_VISIBLE, 0);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    window_ = glfwCreateWindow(800, 800, "Invisible window", nullptr, nullptr);
    if (!window_) {
        mju_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);

    // F-3: a software renderer is ~50x slower and silently kills P-6. Deny by
    // name rather than allow-listing this GPU, which would fail on any other
    // machine that is perfectly fine.
    const GLubyte* renderer_raw = glGetString(GL_RENDERER);
    if (!renderer_raw) {
        mju_error("glGetString(GL_RENDERER) returned null - no current GL context");
    }
    const char* renderer = reinterpret_cast<const char*>(renderer_raw);
    for (const char* bad : kSoftwareGl) {
        if (std::strstr(renderer, bad)) {
            mju_error("software GL, not hardware: matched '%s' in '%s'", bad, renderer);
        }
    }
    printf("GL_RENDERER:  %s\n", renderer);

    // Create the offscreen render context
    mjr_defaultContext(&con_);
    mjr_makeContext(model, &con_, mjFONTSCALE_100);

    // mjr_setBuffer returns void and silently keeps the window buffer when
    // offscreen is unavailable, so the state it mutated has to be read back.
    mjr_setBuffer(mjFB_OFFSCREEN, &con_);
    if (con_.currentBuffer != mjFB_OFFSCREEN) {
        mju_error("offscreen framebuffer not selected: currentBuffer is %d",
                  con_.currentBuffer);
    }

    // A scene whose <global offwidth/offheight> never took effect leaves the
    // offscreen buffer at its 640x480 default, and every frame becomes a crop
    // of the upper-left corner.
    viewport_ = mjr_maxViewport(&con_);
    if (viewport_.width != model->vis.global.offwidth ||
        viewport_.height != model->vis.global.offheight) {
        mju_error("offscreen buffer is %d x %d, scene asked for %d x %d",
                  viewport_.width, viewport_.height,
                  model->vis.global.offwidth, model->vis.global.offheight);
    }

    const int gl_error = mjr_getError();
    if (gl_error) {
        mju_error("OpenGL error 0x%x during context setup", gl_error);
    }

    printf("GL context created successfully, viewport %d x %d\n",
           viewport_.width, viewport_.height);
}

GlContext::~GlContext() {
    // Order matters: mjr_freeContext releases GPU objects and needs the context
    // still current, and glfwTerminate destroys the window that owns it.
    mjr_freeContext(&con_);
    glfwTerminate();
}