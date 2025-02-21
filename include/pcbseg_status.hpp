#pragma once

namespace PCBSEG {
    enum class Status {
        SUCCESS = 0,        // 成功
        ERR_INPUT,         // 输入错误
        ERR_SYSTEM,        // 系统错误
        ERR_LOGIC,         // 逻辑错误
        ERR_PROTECTION,    // 保护错误
        ERR_INITIALIZE,    // 初始化错误
        ERR_TIMEOUT,       // 超时错误
        ERR_FATAL,         // 致命错误
        ERR_UNKNOWN        // 未知错误
    };
} 