#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <stdarg.h>

#include <system_error>

#define LOG(...) __log_info(__VA_ARGS__)
// 使用变参进行LOG的打印。比较推荐的打印log的写法
static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

#endif  //__UTILS_HPP__
