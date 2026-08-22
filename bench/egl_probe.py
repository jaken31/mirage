#RULED OUT WSL RENDERING BECAUSE OF INCOMPATIBLE OPENGL DRIVER, SO THIS IS A HARDWARE EGL PROBE


import ctypes
import os

os.environ["EGL_PLATFORM"] = "surfaceless"  # for Mesa, to avoid X11 dependency

# --- constants, from EGL/egl.h ---
EGL_DEFAULT_DISPLAY = None  # NULL
EGL_VENDOR = 0x3053
EGL_VERSION = 0x3054
EGL_CLIENT_APIS = 0x308D
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_NONE=0x3038
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_BIT = 0x0008
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_ALPHA_SIZE = 0x3021
EGL_DEPTH_SIZE = 0x3025
EGL_STENCIL_SIZE = 0x3026
EGL_CONFIG_CAVEAT = 0x3027
EGL_COLOR_BUFFER_TYPE = 0x303F
EGL_RGB_BUFFER = 0x308E
EGL_OPENGL_API = 0x30A2
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056
GL_VERSION = 0x1F02
GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
# Same config MuJoCo asks for, so a pass here implies a pass there.

# --- C signatures ---
# Declare every function before calling it. ctypes assumes a 32-bit int return,
# which silently truncates 64-bit EGL handles.
libegl = ctypes.CDLL("libEGL.so.1")

libegl.eglGetError.argtypes = []
libegl.eglGetError.restype = ctypes.c_int

libegl.eglGetDisplay.argtypes = [ctypes.c_void_p]
libegl.eglGetDisplay.restype = ctypes.c_void_p

libegl.eglInitialize.argtypes = [ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_int)]
libegl.eglInitialize.restype = ctypes.c_uint

libegl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
libegl.eglQueryString.restype = ctypes.c_char_p


def check(value, what):
    if not value:
        raise SystemExit(f"{what} failed, eglGetError=0x{libegl.eglGetError():x}")


def query(display, name):
    return (libegl.eglQueryString(display, name) or b"").decode()


display = libegl.eglGetDisplay(EGL_DEFAULT_DISPLAY)
check(display, "eglGetDisplay")

major, minor = ctypes.c_int(), ctypes.c_int()
check(libegl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)),
      "eglInitialize")

print(f"EGL version      {major.value}.{minor.value}")
print(f"EGL_VERSION      {query(display, EGL_VERSION)}")
print(f"EGL_VENDOR       {query(display, EGL_VENDOR)}")
print(f"EGL_CLIENT_APIS  {query(display, EGL_CLIENT_APIS)}")

config_attributes = [
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 24,
    EGL_STENCIL_SIZE, 8,
    EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,   # desktop GL; default would be GLES
    EGL_NONE
]

attributeList = (ctypes.c_int * len(config_attributes))(*config_attributes)

libegl.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_void_p), 
                                ctypes.c_int, ctypes.POINTER(ctypes.c_int)]


config = ctypes.c_void_p()

configCount = ctypes.c_int()
config_size = 1 
check(libegl.eglChooseConfig(display, attributeList, ctypes.byref(config), config_size, ctypes.byref(configCount)), "eglChooseConfig")

if configCount.value == 0:
    raise SystemExit("no matching EGL config ")


libegl.eglBindAPI.argtypes = [ctypes.c_uint]
libegl.eglBindAPI.restype = ctypes.c_uint

check(libegl.eglBindAPI(EGL_OPENGL_API), "eglBindAPI")


libegl.eglCreatePbufferSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
libegl.eglCreatePbufferSurface.restype = ctypes.c_void_p

surface_attributes = [
    EGL_WIDTH, 64,
    EGL_HEIGHT, 64,
    EGL_NONE
]

surfaceAttribList = (ctypes.c_int * len(surface_attributes))(*surface_attributes)

surface = libegl.eglCreatePbufferSurface(display, config, surfaceAttribList)

check(surface, "eglCreatePbufferSurface")


libegl.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
libegl.eglCreateContext.restype = ctypes.c_void_p

context_attributes = [
    EGL_NONE
]

context = libegl.eglCreateContext(display, config, None, (ctypes.c_int * len(context_attributes))(*context_attributes))
check(context, "eglCreateContext")

libegl.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
libegl.eglMakeCurrent.restype = ctypes.c_uint

check(libegl.eglMakeCurrent(display, surface, surface, context), "eglMakeCurrent")

libgl = ctypes.CDLL("libGL.so.1")

try: 
    libgl.glGetString
except AttributeError:
    libgl.glGetString = ctypes.CDLL("libOpenGL.so.0").glGetString

SOFTWARE_MARKERS = ("llvmpipe", "softpipe", "swrast", "osmesa")
libgl.glGetString.restype = ctypes.c_char_p
libgl.glGetString.argtypes = [ctypes.c_uint]

gl_version = (libgl.glGetString(GL_VERSION) or b"").decode()
gl_vendor = (libgl.glGetString(GL_VENDOR) or b"").decode()
gl_renderer = (libgl.glGetString(GL_RENDERER) or b"").decode()


if not gl_version:
    raise SystemExit("glGetString returned NULL, no OpenGL context current?")

print(f"OpenGL version   {gl_version}")
print(f"OpenGL vendor    {gl_vendor}")
print(f"OpenGL renderer  {gl_renderer}")

hit = [m for m in SOFTWARE_MARKERS if m in f"{gl_vendor} {gl_renderer}".lower()]
if hit:
    raise SystemExit(f"FAIL software rasterizer ({hit[0]}) - ~50x too slow, P-6 unreachable")

print("PASS hardware EGL")