#pragma once

#include <mujoco/mujoco.h>

// Only a pointer to GLFW's window type is used here, so a forward declaration
// is enough. That keeps the GLFW and OpenGL headers out of every file that
// includes this one; only gl_context.cpp needs them.
struct GLFWwindow;

// Owns the offscreen render target: a hidden GLFW window that holds the GL
// context, MuJoCo's render context built on it, and the viewport size. The
// constructor checks that size against the model's offwidth/offheight, which
// main.cpp sets from config first, so it is the configured render size.
//
// Create exactly one, in main, before anything renders. It does not own the
// model, the scene, or the camera.
class GlContext {
public:
  // Runs the whole setup and aborts via mju_error on any failure, so you never
  // get a half-built context. Needs an already-loaded model because
  // mjr_makeContext sizes its GPU buffers from it.
  explicit GlContext(const mjModel* model);
  ~GlContext();

  // Copying is deleted, not just unused: a copy would share the GPU handles and
  // the second destructor would free them twice. Declaring a destructor already
  // disables the implicit moves, so nothing else needs deleting.
  GlContext(const GlContext&) = delete;
  GlContext& operator=(const GlContext&) = delete;

  // const on purpose. Every render-loop call (mjr_render, mjr_readPixels,
  // mjr_maxViewport) takes a const context. The only call that needs a mutable
  // one, mjr_setBuffer, runs once in the constructor. So no caller can switch
  // rendering back to the window framebuffer, which is the rule this class
  // exists to enforce.
  const mjrContext* context() const { return &con_; }

  // Size of the offscreen buffer, from mjr_maxViewport. Cached so nothing else
  // recomputes it or hardcodes 64.
  mjrRect viewport() const { return viewport_; }

private:
  GLFWwindow* window_ = nullptr;
  // The constructor does the real setup (mjr_defaultContext). Zeroing here only
  // ensures no field is garbage if construction aborts partway.
  mjrContext con_ = {};
  mjrRect viewport_ = {};
};
