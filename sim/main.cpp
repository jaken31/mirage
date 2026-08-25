#include <cstdio>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

int main() {
    printf("C++ version: %ld\n", __cplusplus);
    printf("MuJoCo:      %s\n", mj_versionString());
    printf("GLFW:        %s\n", glfwGetVersionString());
    return 0;
}
