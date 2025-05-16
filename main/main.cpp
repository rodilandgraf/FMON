#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_err.h"
#include "driver/gpio.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_http_client.h"
#include "esp_spiffs.h"
extern "C"
{
#include "MLX90640_API.h"
}
#include "modelo_int8.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// GPIOs e delay pra main
#define GPIO_32 GPIO_NUM_32
#define GPIO_26 GPIO_NUM_26
//const TickType_t xDelay = 50 / portTICK_PERIOD_MS;

// Variáveis MLX90640
#define TA_SHIFT 8
float emissivity = 0.95;
float tr;
unsigned char slaveAddress = 0x33;
static uint16_t mlx90640Frame[834];
paramsMLX90640 mlx90640;
static float mlx90640To[768];
int status;

// Variáveis imagem
#define BMP_HEADER_SIZE 54
#define WIDTH 32
#define HEIGHT 24
#define MAX_CHAR_SIZE 64

static int grayImageInt[768];

typedef struct
{
    uint8_t gray;
} GrayscalePixel;

int16_t minTemp = 20;  
int16_t maxTemp = 300; 

// TensorFlow
constexpr int kTensorArenaSize = 40 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
int fire_value = 0;

tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

void init_tflite_model()
{
    const tflite::Model *model = tflite::GetModel(modelo_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE("TFLite", "Modelo TFLite incompatível!");
        return;
    }

    static tflite::MicroMutableOpResolver<8> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddLogistic();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE("TFLite", "Erro ao alocar tensores!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI("TFLite", "Modelo TFLite inicializado com sucesso.");
}

void run_inference(int *grayImageInt)
{
    if (!interpreter || !input || !output)
    {
        ESP_LOGE("TFLite", "Intérprete não inicializado.");
        return;
    }

    for (int i = 0; i < 32 * 24; i++)
    {
        float input_float = grayImageInt[i] / 255.0f;
        int val = roundf(input_float / input->params.scale) + input->params.zero_point;
        if (val > 127)
            val = 127;
        if (val < -128)
            val = -128;
        input->data.int8[i] = (int8_t)val;
    }

    if (interpreter->Invoke() != kTfLiteOk)
    {
        ESP_LOGE("TFLite", "Erro ao executar inferência.");
        return;
    }

    int8_t sem_fogo = output->data.int8[0];
    int8_t fogo = output->data.int8[1];

    int predicted = (fogo > sem_fogo) ? 1 : 0;
    fire_value = predicted;
    ESP_LOGI("TFLite", "Saída bruta int8 → sem fogo: %d, fogo: %d", sem_fogo, fogo);
    ESP_LOGI("TFLite", "Resultado: %s\n", predicted ? "FOGO DETECTADO" : "SEM FOGO");
}

// SPIFFS para salvar imagem
void init_spiffs()
{
    ESP_LOGI("SPIFFS", "Inicializando SPIFFS");

    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL, 
        .max_files = 5,
        .format_if_mount_failed = true};

    esp_err_t ret = esp_vfs_spiffs_register(&conf);

    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGE("SPIFFS", "Falha ao montar ou formatar o sistema de arquivos");
        }
        else if (ret == ESP_ERR_NOT_FOUND)
        {
            ESP_LOGE("SPIFFS", "Partição SPIFFS não encontrada");
        }
        else
        {
            ESP_LOGE("SPIFFS", "Erro SPIFFS: %s", esp_err_to_name(ret));
        }
        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret == ESP_OK)
    {
        ESP_LOGI("SPIFFS", "Sistema de arquivos montado. Total: %d, Usado: %d", total, used);
    }
    else
    {
        ESP_LOGE("SPIFFS", "Erro ao obter informações: %s", esp_err_to_name(ret));
    }
}

void save_bmp_to_spiffs(const char *filename, uint8_t *data, size_t size)
{
    char filepath[64];
    snprintf(filepath, sizeof(filepath), "/spiffs/%s", filename);
    FILE *f = fopen(filepath, "wb");
    if (f == NULL)
    {
        ESP_LOGE("SPIFFS", "Falha ao abrir arquivo para escrita");
        return;
    }

    fwrite(data, 1, size, f);
    fclose(f);
    ESP_LOGI("SPIFFS", "Imagem salva em: %s (%d bytes)", filepath, size);
}

// Processar imagem GrayScale
void convertToGrayscaleInt(float *temperatureData, int *grayImageInt, int16_t minTemp, int16_t maxTemp)
{
    for (int i = 0; i < 768; i++)
    {
        int16_t temp = temperatureData[i];

        int norm = (temp - minTemp) * 255 / (maxTemp - minTemp);
        if (norm < 0)
            norm = 0;
        if (norm > 255)
            norm = 255;

        grayImageInt[i] = norm; 
    }
}

// Criar BMP
uint8_t *create_bmp_gray(int *gray, int *out_size)
{
    int row_padded = (WIDTH + 3) & (~3);
    int pixel_array_size = row_padded * HEIGHT;
    int file_size = 14 + 40 + 1024 + pixel_array_size;

    *out_size = file_size;
    uint8_t *bmp = (uint8_t *)malloc(file_size);
    if (!bmp)
        return NULL;
    memset(bmp, 0, file_size);

    bmp[0] = 'B';
    bmp[1] = 'M';
    bmp[2] = file_size & 0xFF;
    bmp[3] = (file_size >> 8) & 0xFF;
    bmp[4] = (file_size >> 16) & 0xFF;
    bmp[5] = (file_size >> 24) & 0xFF;

    int offset = 14 + 40 + 1024;
    bmp[10] = offset & 0xFF;
    bmp[11] = (offset >> 8) & 0xFF;
    bmp[12] = (offset >> 16) & 0xFF;
    bmp[13] = (offset >> 24) & 0xFF;

    bmp[14] = 40;
    bmp[18] = WIDTH & 0xFF;
    bmp[19] = (WIDTH >> 8) & 0xFF;
    bmp[22] = HEIGHT & 0xFF;
    bmp[23] = (HEIGHT >> 8) & 0xFF;
    bmp[26] = 1;
    bmp[28] = 8; 
    bmp[34] = pixel_array_size & 0xFF;
    bmp[35] = (pixel_array_size >> 8) & 0xFF;

    for (int i = 0; i < 256; i++)
    {
        int offset = 14 + 40 + i * 4;
        bmp[offset + 0] = i;
        bmp[offset + 1] = i;
        bmp[offset + 2] = i;
        bmp[offset + 3] = 0;
    }

    uint8_t *p = bmp + 14 + 40 + 1024;
    for (int y = HEIGHT - 1; y >= 0; y--)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            *p++ = (uint8_t)gray[y * WIDTH + x];
        }
        int padding = row_padded - WIDTH;
        for (int i = 0; i < padding; i++)
        {
            *p++ = 0;
        }
    }
    return bmp;
}

// Enviar BMP via HTTPS POST
extern "C" {
    extern const uint8_t _binary_lse_pem_start[];
    extern const uint8_t _binary_lse_pem_end[];
}

void send_bmp_https_post(uint8_t *bmp_data, int bmp_size)
{
    const char *boundary = "----ESP32Boundary";
    char content_type[128];
    snprintf(content_type, sizeof(content_type), "multipart/form-data; boundary=%s", boundary);

    esp_http_client_config_t config = {
        .url = "https://lse.cp.utfpr.edu.br/fmon/post-file.php",
        .cert_pem = reinterpret_cast<const char *>(_binary_lse_pem_start),
        .transport_type = HTTP_TRANSPORT_OVER_SSL,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);

    char part1[512];
    snprintf(part1, sizeof(part1),
         "--%s\r\n"
         "Content-Disposition: form-data; name=\"api_key\"\r\n\r\n"
         "Fm#25\r\n"
         "--%s\r\n"
         "Content-Disposition: form-data; name=\"fire\"\r\n\r\n"
         "%d\r\n"
         "--%s\r\n"
         "Content-Disposition: form-data; name=\"file\"; filename=\"image.bmp\"\r\n"
         "Content-Type: image/bmp\r\n\r\n",
         boundary, boundary, fire_value, boundary);

    const char *part3_format = "\r\n--%s--\r\n";
    char part3[64];
    snprintf(part3, sizeof(part3), part3_format, boundary);

    int total_len = strlen(part1) + bmp_size + strlen(part3);

    esp_http_client_set_method(client, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", content_type);
    esp_http_client_set_header(client, "Content-Length", (const char *)NULL);
    esp_http_client_open(client, total_len);

    esp_http_client_write(client, part1, strlen(part1));
    esp_http_client_write(client, (char *)bmp_data, bmp_size);
    esp_http_client_write(client, part3, strlen(part3));

    int status_code = esp_http_client_fetch_headers(client);
    if (status_code > 0)
    {
        ESP_LOGI("HTTP", "Status code: %d", esp_http_client_get_status_code(client));
    }
    else
    {
        ESP_LOGE("HTTP", "Falha ao enviar dados. Código: %d", status_code);
    }
    esp_http_client_cleanup(client);
}

// Captura de imagem
void capture_image(int atual_26, int atual_32, int anterior_26, int anterior_32)
{

    MLX90640_GetSubFrameData(slaveAddress, mlx90640Frame);
    tr = MLX90640_GetTa(mlx90640Frame, &mlx90640) - TA_SHIFT;
    MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, mlx90640To);

    MLX90640_GetSubFrameData(slaveAddress, mlx90640Frame);
    tr = MLX90640_GetTa(mlx90640Frame, &mlx90640) - TA_SHIFT;
    MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, mlx90640To);

    convertToGrayscaleInt(mlx90640To, grayImageInt, minTemp, maxTemp);

    run_inference(grayImageInt);

    int bmp_size = 0;
    uint8_t *bmp_data = create_bmp_gray(grayImageInt, &bmp_size);
    if (bmp_data)
    {
        ESP_LOGI("BMP", "Imagem BMP criada com sucesso (%d bytes)", bmp_size);

        save_bmp_to_spiffs("imagem.bmp", bmp_data, bmp_size);
        send_bmp_https_post(bmp_data, bmp_size);

        free(bmp_data);
    }
    else
    {
        ESP_LOGE("BMP", "Falha ao criar imagem BMP");
    }

    ESP_LOGI("TFLite", "Resultado: %d", fire_value);
    ESP_LOGI("CAMERA", "Matriz de imagem em escala de cinza:");
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            printf("%3d ", grayImageInt[y * WIDTH + x]);
        }
        printf("\n"); 
    }
    printf("\n"); 
}

// WiFi
static const char *TAG = "WiFi";

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
    {
        ESP_LOGI(TAG, "WiFi connecting...");
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        ESP_LOGW(TAG, "WiFi disconnected. Reconnecting...");
        esp_wifi_connect();
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "WiFi connected. Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_connection()
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "Batore",         
            .password = "corinthians2012",
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi initialization complete.");
}

extern "C" void app_main(void)
{
    // Inicializa NVS e WiFi
    nvs_flash_init();
    wifi_connection();

    // Delay para visualização
    vTaskDelay(2000 / portTICK_PERIOD_MS);

    // Inicializa TensorFlow e SPIFFS
    init_tflite_model();
    init_spiffs();

    // inicializa camera
    ESP_LOGI("CAMERA", "Iniciando MLX90640");
    IR_I2CInit();
    uint16_t *eeMLX90640 = (uint16_t *)malloc(832 * sizeof(uint16_t));
    if (eeMLX90640 == NULL)
    {
        ESP_LOGE("ERROR", "Falha ao alocar memória para eeMLX90640");
        return;
    }
    status = MLX90640_DumpEE(slaveAddress, eeMLX90640);
    status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
    free(eeMLX90640); 

    ESP_LOGI("CAMERA", "Pronto para captura");
    // Fim das inicializações

    gpio_set_direction(GPIO_32, GPIO_MODE_INPUT);
    gpio_set_direction(GPIO_26, GPIO_MODE_INPUT);

    int estado_anterior_32 = 0;
    int estado_anterior_26 = 0;

    while (1)
    {
        // Captura da imagem
        int estado_atual_32 = gpio_get_level(GPIO_32);
        int estado_atual_26 = gpio_get_level(GPIO_26);

        if ((estado_atual_26 == 1 && estado_anterior_26 == 0) || (estado_atual_32 == 1 && estado_anterior_32 == 0))
        {
            capture_image(estado_atual_26, estado_atual_32, estado_anterior_26, estado_anterior_32);
        }

        estado_anterior_32 = estado_atual_32;
        estado_anterior_26 = estado_atual_26;

        vTaskDelay(50 / portTICK_PERIOD_MS);
    }
}