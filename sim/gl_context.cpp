#include "gl_context.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <GLFW/glfw3.h>

namespace {
    void GlfwErrorCallback(int error, const char* description) {
        std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
    }
    const char* const kSoftwareGl[] = {
        // Windows
        "GDI Generic",
        "Microsoft Basic Render Driver",
        // Linux: Mesa's CPU rasterizers, e.g. "llvmpipe (LLVM 22.1.8, 256 bits)"
        "llvmpipe",
        "softpipe",
    };

}

GlContext::GlContext(const mjModel* model) {
    glfwSetErrorCallback(GlfwErrorCallback);

#ifdef __linux__
    // On Linux the dataset is rendered on the NVIDIA GPU and nothing else. On a
    // hybrid laptop GLFW gets the integrated GPU by default, which renders 442
    // of 300,000 frames one pixel differently under the same data_hash
    // (runs.jsonl r55). PRIME render offload hands the context to NVIDIA:
    // __NV_PRIME_RENDER_OFFLOAD alone is enough for EGL (GLFW's Wayland
    // backend), and GLX (its X11 backend) also needs the vendor name.
    //
    // Set here rather than in a launch script, so every way of starting the
    // binary gets them, and overwritten rather than defaulted, so a value left
    // in the shell cannot undo the choice. It works because libglvnd reads
    // both when it first picks a vendor, which is inside glfwInit or window
    // creation, after this. The NVIDIA check below catches anything that still
    // gets past, such as __EGL_VENDOR_LIBRARY_FILENAMES naming Mesa.
    setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
#endif

    if (!glfwInit()) {
        mju_error("Failed to initialize GLFW");
    }

    // Nothing is ever drawn to this window; it exists only to own the GL
    // context. Single-buffered because nothing is ever swapped.
    glfwWindowHint(GLFW_VISIBLE, 0);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    window_ = glfwCreateWindow(800, 800, "Invisible window", nullptr, nullptr);
    if (!window_) {
        mju_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);

    // A software renderer is about 50x slower and quietly wrecks the
    // data-generation speed target (500 frames/s). Reject known software
    // renderers by name instead of allowing only this GPU, which would wrongly
    // fail on any other good machine. Linux adds a vendor check further down,
    // for a different reason: which frames come out, not how fast.
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

#ifdef __linux__
    // An allow-list on Linux, on top of the deny-list: real hardware from
    // another vendor passes the deny-list and still renders different bytes.
    // Windows keeps the deny-list alone.
    if (!std::strstr(renderer, "NVIDIA")) {
        mju_error("GL_RENDERER is '%s', not NVIDIA. On Linux the dataset is "
                  "rendered on the NVIDIA GPU only - another GPU gives different "
                  "frames under the same data_hash. The binary already sets PRIME "
                  "offload, so check the NVIDIA driver is loaded and that "
                  "__EGL_VENDOR_LIBRARY_FILENAMES does not point elsewhere",
                  renderer);
    }
#endif

    // GL_VERSION carries the driver version on NVIDIA ("4.6.0 NVIDIA
    // 610.57.04") and on Mesa alike.
    const GLubyte* version_raw = glGetString(GL_VERSION);
    printf("GL_RENDERER:  %s\n", renderer);
    printf("GL_VERSION:   %s\n",
           version_raw ? reinterpret_cast<const char*>(version_raw) : "(null)");

    mjr_defaultContext(&con_);
    mjr_makeContext(model, &con_, mjFONTSCALE_100);

    // mjr_setBuffer returns nothing and silently stays on the window buffer
    // when offscreen is unavailable, so read the result back to check.
    mjr_setBuffer(mjFB_OFFSCREEN, &con_);
    if (con_.currentBuffer != mjFB_OFFSCREEN) {
        mju_error("offscreen framebuffer not selected: currentBuffer is %d",
                  con_.currentBuffer);
    }

    // If offwidth/offheight never took effect, the offscreen buffer stays at
    // its 640x480 default and every frame becomes a crop of the top-left
    // corner. main.cpp sets both from config (not from the XML), so this checks
    // that the driver gave us the size we asked for.
    viewport_ = mjr_maxViewport(&con_);
    if (viewport_.width != model->vis.global.offwidth ||
        viewport_.height != model->vis.global.offheight) {
        mju_error("offscreen buffer is %d x %d, config asked for %d x %d",
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
    // Order matters: mjr_freeContext frees GPU objects and needs the context
    // still current, and glfwTerminate destroys the window that owns it.
    mjr_freeContext(&con_);
    glfwTerminate();
}