#include <cstdio>
#include <cstdlib>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "gl_context.h"

int main(int argc, const char** argv) {
    printf("C++ version: %ld\n", __cplusplus);
    printf("MuJoCo:      %s\n", mj_versionString());
    printf("GLFW:        %s\n", glfwGetVersionString());

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.xml>\n", argv[0]);
        return EXIT_FAILURE;
    }

    constexpr int error_buffer_size = 1000;
    // Initialised, not just declared: if mj_loadXML fails without writing
    // here, printing an uninitialised array walks off the end of it.
    char error_buffer[error_buffer_size] = "Could not load model";
    mjModel* model = mj_loadXML(argv[1], nullptr, error_buffer, error_buffer_size);
    if (!model) {
        mju_error("Failed to load model from '%s': %s", argv[1], error_buffer);
    }

    GlContext context(model);
    mj_deleteModel(model);
    return 0;
}
