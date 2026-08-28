#pragma once

#include <mujoco/mujoco.h>

// GLFW's window type is opaque - it is only ever used through a pointer - so a
// forward declaration is enough here. That keeps <GLFW/glfw3.h> and the OpenGL
// headers out of every file that includes this one; only gl_context.cpp needs
// them.
struct GLFWwindow;

// Owns the offscreen render target: a hidden GLFW window holding the GL
// context, the mjrContext built against it, and the viewport that was checked
// against the model's offwidth/offheight at construction - which main.cpp writes
// from config before constructing this, so it is the configured render size.
//
// Construct exactly one, in main, before anything renders. It does not own the
// model, the scene, or the camera.
class GlContext {
public:
  // Runs the full setup sequence and aborts via mju_error on any failure, so a
  // half-built context is never handed back. Takes the model because
  // mjr_makeContext sizes its GPU buffers from it - so the model must already
  // be loaded.
  explicit GlContext(const mjModel* model);
  ~GlContext();

  // Deleted rather than merely unused: the compiler-generated copy would
  // duplicate the GPU handles, and the second destructor would free them a
  // second time. Declaring the destructor already suppresses the implicit move
  // operations, so there is nothing further to delete.
  GlContext(const GlContext&) = delete;
  GlContext& operator=(const GlContext&) = delete;

  // const on purpose. Every call in the render loop - mjr_render,
  // mjr_readPixels, mjr_maxViewport - takes a const mjrContext*. The one call
  // that needs a mutable one, mjr_setBuffer, runs once inside the constructor.
  // Handing out const therefore makes it impossible for a caller to redirect
  // rendering back to the window framebuffer, which is the invariant this class
  // exists to hold.
  const mjrContext* context() const { return &con_; }

  // Size of the offscreen buffer, from mjr_maxViewport. Cached so nothing
  // downstream recomputes it or hardcodes 64.
  mjrRect viewport() const { return viewport_; }

private:
  GLFWwindow* window_ = nullptr;
  // mjr_defaultContext in the constructor is the real initialisation; zeroing
  // here only guarantees no member is garbage if construction aborts partway.
  mjrContext con_ = {};
  mjrRect viewport_ = {};
};
