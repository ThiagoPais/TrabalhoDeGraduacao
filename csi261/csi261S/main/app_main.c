/* >>>> csi261S <<<< Startover Version 3.Jan.2023

 esp_now (mac p2p): 2 Masters (M1,M2), 6 Sensors, 1 CSI Collector (M1)
 
    Based on esp-csi/examples/get-started/csi_recv - ESPRESSIF
    @2022 Adolfo Bauchspiess - ITIV/KIT

    Unless required by applicable law or agreed to in writing, this
    software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
    CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include <sys/time.h>

#define CONFIG_LESS_INTERFERENCE_CHANNEL    11
#define CONFIG_SEND_FREQUENCY               1000
//#define configTICK_RATE_HZ                  1000

typedef struct struct_msg { // Sensor Msg Sended
    uint8_t  idSens;
    uint32_t sCount;  // General counter (marker) for all data packages
    uint32_t bc1;
    uint32_t bc2;
    uint32_t tstamp1;  // timestamp of M1 CSI acquisition
    uint32_t tstamp2;  // timestamp of M2 CSI acquisition
    uint32_t tstampS;  // send to M1 timestamp (32 bit in us)
    int8_t   RSSI1;   // from Master 1
    int8_t   RSSI2;   // frin Master 2
    uint8_t  retry;
    int8_t  data[216]; // = 24 + 4 + 216 = 244 bytes
} struct_msg;
struct_msg myMsg;

typedef struct struct_bc {  // Broadcast Msg Received
    uint8_t  idMaster; // origin of bcMsg
    uint8_t  debugFlag; // 0b0000 0001 dbug(M1), 0b0000 0010 dbug(M2), 0b1111 1100 dbug(S:6)
    uint16_t delta;
    uint32_t bcCount;
} struct_bc;
struct_bc bcM1, bcM2;


struct timeval tv_now;
int64_t time_us; // absolute measured
int64_t time_st; // start of running (or reset by M1 "0" gCount)
int64_t time_cb;  // time elapsed since send_cb
int32_t dtime;

int32_t ts2, tsS,LStamp1=0,LStamp2=1, LStamp=0;


esp_err_t ret;


uint8_t myMAC[6];
int8_t seqSens = 0;
uint8_t sendNow = 1; // Received req. from Master; CSI ready in CallBack function


static const char *TAG = "csi_recv";
static const uint8_t CONFIG_CSI_SEND_MAC[] = {0xc4, 0xde, 0xe2, 0xc0, 0x10, 0xc8}; // "8"

/* ITIV Sensors *
uint8_t CONFIG_CSI_SEND_MAC1[] = {0xc0, 0x49, 0xef, 0x4b, 0x2a, 0x14}; // "1"
uint8_t CONFIG_CSI_SEND_MAC2[] = {0xc0, 0x49, 0xef, 0x4b, 0xa2, 0x98}, // "2"
uint8_t CONFIG_CSI_SEND_MAC3[] = {0xc4, 0xde, 0xe2, 0xc0, 0x0a, 0x2c}, // "3"
uint8_t CONFIG_CSI_SEND_MAC4[] = {0xc4, 0xde, 0xe2, 0xc0, 0x09, 0x78}, // "4"
uint8_t CONFIG_CSI_SEND_MAC5[] = {0xc0, 0x49, 0xef, 0x4a, 0xfa, 0x1c}, // "5"
uint8_t CONFIG_CSI_SEND_MAC6[] = {0xc4, 0xde, 0xe2, 0xc0, 0x0a, 0xb8}, // "6"
uint8_t CONFIG_CSI_SEND_MAC7[] = {0xc4, 0xde, 0xe2, 0xc0, 0x10, 0x90}, // "7"
uint8_t CONFIG_CSI_SEND_MAC8[] = {0xc4, 0xde, 0xe2, 0xc0, 0x10, 0xc8}, // "8"

* UnB Sensors *
                                                    84:cc:a8:5e:62:60 // "9"
                                                    f0:08:d1:d2:b4:90 // "10"
* GDH
uint8_t Receiver_Address0[] = {0x0c, 0xb8, 0x15, 0xcc, 0x92, 0x94}; // "11"
uint8_t Receiver_Address1[] = {0x0c, 0xb8, 0x15, 0xf7, 0x8a, 0x60}; // "12"
uint8_t Receiver_Address2[] = {0x0c, 0xb8, 0x15, 0xcd, 0xb9, 0x30}; // "13"
uint8_t Receiver_Address3[] = {0x0c, 0xb8, 0x15, 0xcd, 0x02, 0x7c}; // "14"
uint8_t Receiver_Address3[] = {0x0c, 0xb8, 0x15, 0xf7, 0x90, 0xb0}; // "15"
*/

#define DEC_PLACE_MULT 1000

void print_dtime() {
    gettimeofday(&tv_now, NULL); //��ȡ��ǰ��ʱ�䣬���Ὣ��ǰ��������΢�����洢��tv_now���timeval�ṹ���С�
    time_us = (int64_t)tv_now.tv_sec * 1000000L + (int64_t)tv_now.tv_usec; //�����΢��ת��Ϊ΢�룬���洢��time_us�С�
    dtime = time_us-time_cb;  // dtime in us ������ϴμ�¼��ʱ�䣨time_cb�������ڵ�ʱ�䣨time_us���Ĳ�ֵ����λ��΢�롣�����ֵ�����˴��ϴλ�ȡʱ�䵽���ڵ�ʱ������Ҳ����delta time�����dtime��

//    ets_printf("dtime(us)=%u ",dtime);
    ets_printf(" dt(ms): %d.%02u",  dtime / DEC_PLACE_MULT, abs(dtime) % DEC_PLACE_MULT); //��ets_printf������dtime��ӡ��������λ�Ǻ��롣
}
void print_stime() {
    gettimeofday(&tv_now, NULL);
    time_us = (int64_t)tv_now.tv_sec * 1000000L + (int64_t)tv_now.tv_usec;
    dtime = time_us-time_st;  // dtime in us
    time_cb = time_us;

//    ets_printf("dtime(us)=%u ",dtime);
    ets_printf(" t(ms): %d.%02u",  dtime / DEC_PLACE_MULT, abs(dtime) % DEC_PLACE_MULT);
}

uint8_t na4MAC(uint8_t *mac) {
    uint8_t name;
    uint8_t mac4=mac[4], mac5=mac[5];
    
    if (mac5==0x14) name=1;
    else if (mac5==0x98) name=2; // ITIV
    else if (mac5==0x2c) name=3;
    else if (mac5==0x78) name=4;
    else if (mac5==0x1c) name=5;
    else if (mac5==0xb8) name=6;
    else if (mac5==0x90) {name=7; if (mac4==0xb4) name=10;}
    else if (mac5==0xc8) name=8;
    
    else if (mac5==0x94) name=11; // GDH
    else if (mac5==0x60) {name=12;  if (mac4==0x62) name=9;}
    else if (mac5==0x30) name=13;
    else if (mac5==0x7c) name=14;
    else if (mac5==0xb0) name=15;
    else name=9;
    return name;
}
static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, WIFI_BW_HT40));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_config_espnow_rate(ESP_IF_WIFI_STA, WIFI_PHY_RATE_MCS0_SGI));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}


/* ***********************************************
   RED ��WiFiоƬ���յ�CSI����ʱ����������ͻᱻ����
   ***********************************************/
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) //���յ�����������ָ��ctx��infoָ��
{
    if (!info || !info->buf || !info->mac) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG)); //�������info��info->buf��info->mac�Ƿ�Ϊnull�������null�����ӡһ��������Ϣ���˳�����
        return;
    }
    
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl; //������ȡ���տ�����Ϣ
    LStamp=rx_ctrl->timestamp;//����ʱ�����LStamp����


    if (info->mac[5] == 0xc8 && LStamp > LStamp1 && bcM1.idMaster == 1) { // M136 - Collect  �������MAC��ַ�����һ���ֽ��Ƿ�Ϊ0xc8��ʱ����Ƿ����LStamp1��bcM1.idMaster�Ƿ�Ϊ1

        myMsg.tstamp1 = rx_ctrl->timestamp - ts2;
        myMsg.RSSI1 = rx_ctrl->rssi;
        memcpy(&myMsg.data[0],&info->buf[12],108);
        
        LStamp1=LStamp;
        myMsg.bc1++;

        ets_printf("\n\r\033[0;31mCSI%u bcM1:%u, bc1:%u, bc2:%u",bcM1.idMaster,bcM1.bcCount,myMsg.bc1,myMsg.bc2);
        print_dtime();
    }
    
    else if (info->mac[5] ==0x98 && LStamp > LStamp2) { // M2 "2"

        ESP_ERROR_CHECK(esp_wifi_set_csi(false)); //* No more CSI, until this myMsg is transmitted������esp_wifi_set_csi(false)���ر�CSI

        myMsg.tstamp2 = rx_ctrl->timestamp - ts2;
        myMsg.RSSI2 = rx_ctrl->rssi;
        memcpy(&myMsg.data[108],&info->buf[12],108);
        
        LStamp2=LStamp;
        myMsg.bc2++;

        ets_printf("\n\r\033[0;31mCSI2 bcM2:%u, bc1:%u, bc2:%u",bcM2.bcCount,myMsg.bc1,myMsg.bc2);
        print_dtime();
        ets_printf("           ===>sendNow ~CSI\033[0m ");
        
        sendNow = 1; // Enable Transmission of myMsg by main() to the master M1("8") ������������Է���myMsg��
    }
    else  { // Register "others"
        ets_printf("\n\r\033[0;31mCSI from %u == Spurious!! \033[0m",na4MAC(info->mac));
        print_dtime();
     }
}

/* It is of NO USE - master should process myMsg!!
 *************************************************/
/* ***********************************************
 �� ESP-NOW ������Ϣ�󣬻ᴥ���˺���
   ***********************************************/
static void now_send_cb(const uint8_t *mac, esp_now_send_status_t status) {
    if (status != 0) {
        myMsg.retry++;
    }
    else myMsg.retry=0;   // send was OK!!

//    if (bcM1.debugFlag & 0b100) {
        ets_printf("\n\r\033[0;36m  send_cb() bcC=%u, sC=%u      ",bcM1.bcCount, myMsg.sCount);
        print_dtime();
        ets_printf(" retry=%u, status=%u\033[0m", myMsg.retry, status);
//    }
}


/*
 static void now_recv_cb(const uint8_t *mac, const uint8_t *data, int len) {
    ets_printf("len %d",len);
    uint32_t gCount=data[0]+(data[1]<<8)+(data[2]<<16)+(data[3]<<24);
    ets_printf("Now asking for CSI, gCount: %d\n\r",gCount);
}
*/

/* ***********************************************
 YELLOW �� ESP-NOW �յ���Ϣʱ�������˺�����
   ***********************************************/
static void now_recv_cb(const uint8_t *mac, const uint8_t *incoming, int len) {

    memcpy(&bcM1,incoming, sizeof(bcM1));
    memcpy(&bcM2,incoming, sizeof(bcM2));
    
    if (bcM1.debugFlag & 0b100) {      //�������bcM1��debugFlag�еĵ���λ�Ƿ�Ϊ1
            ets_printf("\n\n\r\033[0;33m   Rx%u now_recv_cb bcC1:%u,              \033[0m", bcM1.idMaster,bcM1.bcCount);
            print_dtime();
        }

    if (bcM1.bcCount == 0) {
        myMsg.sCount = 0;
        myMsg.bc1=0;
        myMsg.bc2=0;

        print_stime();
        time_st = time_us;
        ts2 = time_us;
        tsS = time_us;
    }
    
    if (bcM1.idMaster == 1 ) { // "D8" Enable CSI
                    
        ets_printf("\n\n\r   BC1:%u, S%u, delta:%u, seqSens:%u", bcM1.bcCount,myMsg.idSens,bcM1.delta,seqSens);
        print_stime(); // Start new cycle - (time_cb=time_us)
       
        ets_printf(" END =>NEW CYCLE\033[0;31m CSI\033[0m");
    //����CSI��
        ESP_ERROR_CHECK(esp_wifi_set_csi(true));  // Even bcCount => Enable CSI
    }

    if (bcM1.idMaster == 136) { // "136" Collect BC1
        if (bcM1.debugFlag & 0b100) {
            ets_printf("\n\r\033[0;33m   BC136: %u, CYCLE END: CSI", bcM1.bcCount);
            print_dtime();
            ets_printf("\033[0m");
        }
    }
    
    if (bcM1.idMaster == 2) { // "2" Collect BC2
        if (bcM1.debugFlag & 0b100) {
            ets_printf("\n\r\033[0;33m   BC2: %u,                         ", bcM1.bcCount);
            print_dtime();
            ets_printf("\033[0m");
        }
    }
}

/* ***********************************************
 
   ***********************************************/
static void wifi_csi_init()
{
    
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(false));
    // ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    // ESP_ERROR_CHECK(esp_wifi_set_promiscuous_rx_cb(g_wifi_radar_config->wifi_sniffer_cb));
    
    /**< 20 MHz 128 bytes CSI config */
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = false,
        .channel_filter_en = false,
        .manu_scale        = false,
        .shift             = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    
    ESP_ERROR_CHECK(esp_now_register_recv_cb(now_recv_cb));
    ESP_ERROR_CHECK(esp_now_register_send_cb(now_send_cb));
    
    //30dec22
    //    ESP_ERROR_CHECK(esp_wifi_set_csi(false));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

}

/* ***********************************************
 
   ***********************************************/
void app_main()
{
    //Initialize NVS �����Գ�ʼ��NVS��Non-volatile storage��������洢�ռ����������°汾������ִ��nvs_flash_erase()�����NVS��Ȼ���ٽ��г�ʼ��
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    //Ȼ���ʼ��WiFi��
    wifi_init();
    
    // *******************************
    // ******************************* ���ų�ʼ��ESP-NOW������PMK��Ԥ������Կ��
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    //��������ESP-NOW�ĶԵ���(peer)���ֱ������������ͬ��MAC��ַ��
    esp_now_peer_info_t peerM1 = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xc4, 0xde, 0xe2, 0xc0, 0x10, 0xc8}, // "8"
    };

    esp_now_peer_info_t peerM2 = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xc0, 0x49, 0xef, 0x4b, 0xa2, 0x98},  //"2"
    };
    //���Ž��������Ե������ӵ�ESP-NOW�ĶԵ����б���
    ESP_ERROR_CHECK(esp_now_add_peer(&peerM1));
    ESP_ERROR_CHECK(esp_now_add_peer(&peerM2));

    ESP_LOGI(TAG, "================ CSI SEND ================");
    ESP_LOGI(TAG, "wifi_channel: %d, send_frequency: %d, mac: " MACSTR,
             CONFIG_LESS_INTERFERENCE_CHANNEL, CONFIG_SEND_FREQUENCY, MAC2STR(CONFIG_CSI_SEND_MAC));

    wifi_csi_init(); //��ʼ��CSI��Channel State Information����������漰����ʼ��WiFiоƬ��ĳЩ���ܣ��Ա���պʹ���CSI���ݡ�
    
    
    for (uint8_t i=0;i<214;i++) myMsg.data[i]=0; //Prepare Message Data: Ȼ�����myMsg.data���飬��ȡ�����豸��MAC��ַ��������MAC��ַ�����һ���ֽڸ�myMsg.idSens��ֵ����ͬ��ֵ��Ӧ��ͬ�Ĵ�����
    
    ESP_ERROR_CHECK(esp_base_mac_addr_get(myMAC));
    ets_printf("\n\r idMaster:%u, MACmaster: " MACSTR, myMsg.sCount,MAC2STR(peerM1.peer_addr));
    ets_printf(" myMAC: " MACSTR ,MAC2STR(myMAC));

    
        
    if (myMAC[5]==0x14) myMsg.idSens=1;
    else if (myMAC[5]==0x98) myMsg.idSens=2; // ITIV
    else if (myMAC[5]==0x2c) myMsg.idSens=3;
    else if (myMAC[5]==0x78) myMsg.idSens=4;
    else if (myMAC[5]==0x1c) myMsg.idSens=5;
    else if (myMAC[5]==0xb8) myMsg.idSens=6;
    else if (myMAC[5]==0x90) {myMsg.idSens=7; if (myMAC[4]==0xb4) myMsg.idSens=10;}
    else if (myMAC[5]==0xc8) myMsg.idSens=8;
    
    else if (myMAC[5]==0x94) myMsg.idSens=11; // GDH
    else if (myMAC[5]==0x60) {myMsg.idSens=12;  if (myMAC[4]==0x62) myMsg.idSens=9;}
    else if (myMAC[5]==0x30) myMsg.idSens=13;
    else if (myMAC[5]==0x7c) myMsg.idSens=14;
    else if (myMAC[5]==0xb0) myMsg.idSens=15;
    else myMsg.idSens=9;

    if      (myMsg.idSens==3)  seqSens = 0;  // leave 1 TDMA slot for BC8
    else if (myMsg.idSens==4)  seqSens = 1;
    else if (myMsg.idSens==5) seqSens = 2;
    else if (myMsg.idSens==6) seqSens = 3;
    else if (myMsg.idSens==7) seqSens = 4;
    else if (myMsg.idSens==1) seqSens = 5;
    else                       seqSens = 6;

    myMsg.sCount=0;
    myMsg.bc1=0;
    myMsg.bc2=0;
    
    gettimeofday(&tv_now, NULL);
    time_st = (int64_t)tv_now.tv_sec * 1000000L + (int64_t)tv_now.tv_usec;
    print_stime();
        
    for (; ;) {                 // infinite loop����TDMA����
        if (sendNow) {
            sendNow = 0;        // esp_send_now == ESP_OK// ready for a new CYCLE
            
            // TDMA Wait bcMsg.delta*(0:5) ms. Time Division Sensor Tx >> Try to Avoid Collision
            //  Sensors 3,9,11,12,13,15 ==> wait 0, delta, 2*delta, ...ms
            //  bcM1.delta*seqSens; 0:5 Sensors reload the specific TDMA value
            vTaskDelay(bcM1.delta*seqSens); // TDMA

            myMsg.sCount++;
            myMsg.retry = 0;
            
            if (bcM1.debugFlag & 0b100) {
                ets_printf("\n\r\033[0;34m S%u TDMA wait, bcM1:%u,",myMsg.idSens,   bcM1.bcCount);
                print_dtime();
                ets_printf(", bc1:%u, bc2:%u, ",myMsg.bc1,myMsg.bc2);
                myMsg.tstampS = dtime; // - tsS;
            }
            else {
                gettimeofday(&tv_now, NULL);
                time_us = (int64_t)tv_now.tv_sec * 1000000L + (int64_t)tv_now.tv_usec;
                dtime = time_us-time_cb;  // dtime in us
                
                myMsg.tstampS = dtime; // - tsS;
            }
            
            //          vTaskDelay(1); // give 1 "tick"
            
            esp_err_t ret = esp_now_send(peerM1.peer_addr, (uint8_t *) &myMsg, sizeof(myMsg));
            if (ret != ESP_OK) ESP_LOGW(TAG, "<%s> ESP-NOW send error", esp_err_to_name(ret));
            
            vTaskDelay(3); // give 1 "tick"
            if (bcM1.debugFlag & 0b100) {
                ets_printf("\n\r\033[0;34m   SEND %u main() sC:%u, delta:%u, seq:%u,", myMsg.idSens,myMsg.sCount,bcM1.delta, seqSens);
                print_dtime();
                ets_printf(", debugFlag:%u, retry:%u\n\r\033[0m",bcM1.debugFlag,myMsg.retry);
            }
        }
    }
}
