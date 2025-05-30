#include <stdio.h>
#include <string.h>
#include <microhttpd.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <time.h>
#include <pthread.h>
#include <nvml.h>

// Define nvml functions
typedef nvmlReturn_t (*nvmlInitFunc)();
typedef const char* (*nvmlErrorStringFunc)(nvmlReturn_t);
typedef nvmlReturn_t (*nvmlDeviceGetCountFunc)(unsigned int *);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndexFunc)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*nvmlDeviceGetPowerUsageFunc)(nvmlDevice_t, unsigned int *);
typedef nvmlReturn_t (*nvmlDeviceGetTemperatureFunc)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRatesFunc)(nvmlDevice_t, nvmlUtilization_t *);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfoFunc)(nvmlDevice_t, nvmlMemory_t *);

// struct that include variables and functions
typedef struct {
    void* handle;
    nvmlInitFunc init;
    nvmlErrorStringFunc errorString;
    nvmlDeviceGetCountFunc getCount;
    nvmlDeviceGetHandleByIndexFunc getHandleByIndex;
    nvmlDeviceGetPowerUsageFunc getPowerUsage;
    nvmlDeviceGetTemperatureFunc getTemperature;
    nvmlDeviceGetUtilizationRatesFunc getUtilizationRates;
    nvmlDeviceGetMemoryInfoFunc getMemoryInfo;
    
    unsigned int deviceCount;
} NVMLContext;

static NVMLContext *g_ctx = NULL;
//initiate global variable
typedef struct{
    int powerUsage;     // mW
    int temperature;    // C
    int utilGpu;        // %
    int utilMem;        // %
    long long memTotal; // MB
    long long memUsed;  // MB
    long long memFree;  // MB
} GpuMetrics;

volatile GpuMetrics gpuMetrics = {0};
// Function that initiate nvml
NVMLContext* initNVML() {
    NVMLContext* ctx = malloc(sizeof(NVMLContext));
    if (!ctx) {
        perror("malloc");
        return NULL;
    }

    ctx->handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!ctx->handle) {
        printf("Failed to load libnvidia-ml.so.1: %s\n", dlerror());
        free(ctx);
        return NULL;
    }
    dlerror(); // Delete error before using dlsys

    ctx->init = (nvmlInitFunc)dlsym(ctx->handle, "nvmlInit_v2");
    ctx->errorString = (nvmlErrorStringFunc)dlsym(ctx->handle, "nvmlErrorString");
    ctx->getCount = (nvmlDeviceGetCountFunc)dlsym(ctx->handle, "nvmlDeviceGetCount_v2");
    ctx->getHandleByIndex = (nvmlDeviceGetHandleByIndexFunc)dlsym(ctx->handle, "nvmlDeviceGetHandleByIndex_v2");
    ctx->getPowerUsage = (nvmlDeviceGetPowerUsageFunc)dlsym(ctx->handle, "nvmlDeviceGetPowerUsage");
    ctx->getTemperature = (nvmlDeviceGetTemperatureFunc)dlsym(ctx->handle, "nvmlDeviceGetTemperature");
    ctx->getUtilizationRates = (nvmlDeviceGetUtilizationRatesFunc)dlsym(ctx->handle, "nvmlDeviceGetUtilizationRates");
    ctx->getMemoryInfo = (nvmlDeviceGetMemoryInfoFunc)dlsym(ctx->handle, "nvmlDeviceGetMemoryInfo");

    if (!ctx->init || !ctx->errorString || !ctx->getCount || 
        !ctx->getHandleByIndex || !ctx->getPowerUsage ||
        !ctx->getTemperature || !ctx->getUtilizationRates || !ctx->getMemoryInfo) {
        printf("Failed to load NVML functions: %s\n", dlerror());
        dlclose(ctx->handle);
        free(ctx);
        return NULL;
    }

    nvmlReturn_t result = ctx->init();
    if (result != NVML_SUCCESS) {
        printf("Fail to initialize NVML: %s\n", ctx->errorString(result));
        dlclose(ctx->handle);
        free(ctx);
        return NULL;
    }

    result = ctx->getCount(&(ctx->deviceCount));
    if (result != NVML_SUCCESS) {
        printf("Cannot get the number of GPUs: %s\n", ctx->errorString(result));
        dlclose(ctx->handle);
        free(ctx);
        return NULL;
    }

    printf("NVML initialized successfully. Number of GPUs: %u\n", ctx->deviceCount);
    return ctx;
}

void cleanupNVML(NVMLContext* ctx) {
    if (ctx) {
        if (ctx->handle)
            dlclose(ctx->handle);
        free(ctx);
    }
}
// Func get sample and print: time, power usage, temperature, utilization and memory info.
void sampleNVMLData(NVMLContext* ctx){
    time_t now = time(NULL);
    printf("Timestamp: %s", ctime(&now));
    
    for (unsigned int i = 0; i < ctx->deviceCount; i++){
        nvmlDevice_t device;
        if (ctx->getHandleByIndex(i, &device) != NVML_SUCCESS) continue;

        // (1) Lấy chỉ số GPU chung như trước...
        unsigned int powerUsage, temp;
        nvmlUtilization_t utilization;
        nvmlMemory_t memory;
        ctx->getPowerUsage(device, &powerUsage);
        ctx->getTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        ctx->getUtilizationRates(device, &utilization);
        ctx->getMemoryInfo(device, &memory);

        // Cập nhật vào global struct
        gpuMetrics.powerUsage = powerUsage;
        gpuMetrics.temperature = temp;
        gpuMetrics.utilGpu = utilization.gpu;
        gpuMetrics.utilMem = utilization.memory;
        gpuMetrics.memTotal = memory.total / (1024ULL * 1024ULL);
        gpuMetrics.memUsed  = memory.used  / (1024ULL * 1024ULL);
        gpuMetrics.memFree  = memory.free  / (1024ULL * 1024ULL);

        // In ra console (tuỳ bạn)
        printf("GPU %u: Power %u mW, Temp %u C, Util GPU %u%%, Mem Used %llu/%llu MB\n",
            i, powerUsage, temp, utilization.gpu,
            gpuMetrics.memUsed, gpuMetrics.memTotal);

        // (2) Lấy danh sách tiến trình compute đang chạy
        unsigned int procCount = 0;
        //  gọi lần 1 để biết số tiến trình
        nvmlReturn_t err = nvmlDeviceGetComputeRunningProcesses(device, &procCount, NULL);
        if (err == NVML_ERROR_INSUFFICIENT_SIZE && procCount > 0) {
            nvmlProcessInfo_t *procs = malloc(procCount * sizeof(*procs));
            if (nvmlDeviceGetComputeRunningProcesses(device, &procCount, procs) == NVML_SUCCESS) {
                for (unsigned int j = 0; j < procCount; ++j) {
                    printf("  → PID %u uses %llu bytes GPU memory\n",
                        procs[j].pid, procs[j].usedGpuMemory);
                }
            }
            free(procs);
        }
        printf("\n");
    }
}

volatile unsigned int global_power_usage = 0;  // ví dụ cho GPU 0

// Func callbackk for HTTPS result
static enum MHD_Result handle_request(void *cls,
    struct MHD_Connection *connection,
    const char *url,
    const char *method,
    const char *version,
    const char *upload_data,
    size_t *upload_data_size,
    void **con_cls)
{
    if (strcmp(method, "GET") != 0 || strcmp(url, "/metrics") != 0)
        return MHD_NO;

    const size_t BUF_SIZE = 16384;
    char *response = malloc(BUF_SIZE);
    if (!response)
        return MHD_NO;
    size_t len = 0;

    // 2a. GPU‐wide metrics (đã có)

    len += snprintf(response + len, BUF_SIZE - len,
        "# HELP gpu_process_memory_bytes GPU memory used per process (bytes)\n"
        "# TYPE gpu_process_memory_bytes gauge\n"
    );

    // 2b. Lấy handle GPU từ g_ctx
    for (unsigned int i = 0; i < g_ctx->deviceCount; ++i) {
        nvmlDevice_t device;
        if (g_ctx->getHandleByIndex(i, &device) != NVML_SUCCESS)
            continue;

        // (1) hỏi số tiến trình
        unsigned int procCount = 0;
        nvmlReturn_t err = nvmlDeviceGetComputeRunningProcesses(device, &procCount, NULL);
        if (err != NVML_ERROR_INSUFFICIENT_SIZE || procCount == 0)
            continue;

        // (2) cấp mảng và gọi lại để lấy chi tiết
        nvmlProcessInfo_t *procs = malloc(procCount * sizeof(*procs));
        if (!procs)
            continue;
        err = nvmlDeviceGetComputeRunningProcesses(device, &procCount, procs);
        if (err == NVML_SUCCESS) {
            for (unsigned int j = 0; j < procCount; ++j) {
                len += snprintf(response + len, BUF_SIZE - len,
                    "gpu_process_memory_bytes{gpu=\"%u\",pid=\"%u\"} %llu\n",
                    i,
                    procs[j].pid,
                    procs[j].usedGpuMemory
                );
            }
        }
        free(procs);
    }

    // 3. Trả HTTP response
    struct MHD_Response *resp = MHD_create_response_from_buffer(
        len, response, MHD_RESPMEM_MUST_FREE
    );
    enum MHD_Result ret = MHD_queue_response(connection, MHD_HTTP_OK, resp);
    MHD_destroy_response(resp);
    return ret;
}


#define PORT 8080

// Func run HTTP server
void start_http_server() {
    struct MHD_Daemon *daemon;
    daemon = MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD, PORT,
                              NULL, NULL, &handle_request, NULL,
                              MHD_OPTION_END);
    if (!daemon) {
        fprintf(stderr, "Failed to start HTTP server.\n");
        exit(EXIT_FAILURE);
    }
    printf("HTTP server is running on port %d\n", PORT);
    
    // until get Ctrl+C
    getchar();
    MHD_stop_daemon(daemon);
}

int main(void) {
    NVMLContext *ctx = initNVML();
    if (!ctx)
        return 1;
    g_ctx = ctx;   // <-- lưu context toàn cục
    
    // Create a stream to run HTTP server exporter
    pthread_t server_thread;
    if(pthread_create(&server_thread, NULL,(void*(*)(void*))start_http_server,NULL)!=0){
        perror("pthread_create");
        cleanupNVML(ctx);
        return 1;
    }
    // Get sample every second
    while (1) {
        sampleNVMLData(ctx);
        sleep(1);
    }
    cleanupNVML(ctx);
    return 0;
}
