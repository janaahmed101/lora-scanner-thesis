import json
import os


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "lora_sdr.json")
CONFIG_PATH = os.environ.get("LORA_SDR_CONFIG_PATH", _DEFAULT_CONFIG_PATH)


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


_CFG = _load_json(CONFIG_PATH)


def _get(path, default):
    node = _CFG
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# Debug (single toggle with backward-compatible fallback)
_DEBUG_ENABLE = _get(("debug", "enable"), None)
if _DEBUG_ENABLE is None:
    _DEBUG_ENABLE = bool(_get(("debug", "lorawan_trace"), True)) or bool(_get(("debug", "frame_dump"), True))
DEBUG_ENABLE = bool(_DEBUG_ENABLE)
LORAWAN_DEBUG = DEBUG_ENABLE
LORA_DEBUG_FRAME_DUMP = DEBUG_ENABLE

# Radio (shared)
RADIO_FREQUENCY_HZ = int(_get(("radio", "frequency_hz"), 868100000))
RADIO_BANDWIDTH_HZ = int(_get(("radio", "bandwidth_hz"), 125000))
RADIO_SF = int(_get(("radio", "spreading_factor"), 8))
RADIO_CR = int(_get(("radio", "coding_rate"), 5))
RADIO_OVERSAMPLE = int(_get(("radio", "oversample"), 2))
RADIO_SAMPLE_RATE_HZ = int(RADIO_BANDWIDTH_HZ * RADIO_OVERSAMPLE)

# TX bridge
LORA_TX_DEV_ID = int(_get(("tx", "device_id"), 0))
LORA_TX_USRP_ADDR = str(_get(("tx", "usrp_addr"), "10.44.0.6"))
LORA_TX_RECV_BUFF_SIZE = int(_get(("tx", "recv_buffer_size"), 307200))
LORA_TX_SEND_BUFF_SIZE = int(_get(("tx", "send_buffer_size"), 307200))
LORA_TX_CRC = int(_get(("tx", "crc"), 1))
LORA_TX_OVERSAMPLING = int(_get(("tx", "oversampling"), 2))
LORA_TX_PREAMBLE_LEN = int(_get(("tx", "preamble_len"), 8))
LORA_TX_GAIN = float(_get(("tx", "gain"), 20.0))
LORA_TX_NUM_PKTS = int(_get(("tx", "num_pkts"), 0))
LORA_TX_MIN_LEAD_S = float(_get(("tx", "min_lead_s"), 0.020))
LORA_TX_PREINIT = bool(_get(("tx", "preinit"), True))

# RX bridge
LORA_RX_SYNC1 = str(_get(("rx", "sync_word", "sync1"), "0x34"))
LORA_RX_SYNC2 = str(_get(("rx", "sync_word", "sync2"), "0x12"))
LORA_RX_CAPTURE_TIME = float(_get(("rx", "capture_time_s"), 1.0))
LORA_RX_OVERSAMPLE = int(_get(("rx", "oversample"), RADIO_OVERSAMPLE))
LORA_RX_PREAMBLE_THRESH = int(_get(("rx", "preamble_thresh"), 5))

# SDRPlay
SDRPLAY_URL = str(_get(("sdrplay", "url"), "http://10.44.0.7:1234/"))

# Rec_Stream defaults
RS_SDR_TYPE = str(_get(("rec_stream", "sdr_type"), "usrp")).lower()
RS_GAIN = float(_get(("rec_stream", "gain"), 60.0))
RS_CENTER_FREQ_HZ = int(_get(("rec_stream", "center_freq_hz"), RADIO_FREQUENCY_HZ))
RS_RX_STATE = bool(_get(("rec_stream", "rx_state"), True))

# Realtime receiver
RT_RX_TYPE = str(_get(("rt_receiver", "rx_type"), "usrp")).lower()
RT_TOTAL_RX_TIME_S = float(_get(("rt_receiver", "total_rx_time_s"), 500.0))
RT_RUN_FOREVER = bool(_get(("rt_receiver", "run_forever"), False))
RT_CAPTURE_TIME_S = float(_get(("rt_receiver", "capture_time_s"), 0.25))
RT_LOCALIZATION = bool(_get(("rt_receiver", "localization"), False))
RT_ENABLE_SERVER_BRIDGE = bool(_get(("rt_receiver", "enable_server_bridge"), True))
RT_ENABLE_DOWNLINK_TX = bool(_get(("rt_receiver", "enable_downlink_tx"), True))
RT_BW_HZ = int(_get(("rt_receiver", "bw_hz"), RADIO_BANDWIDTH_HZ))
RT_SF = int(_get(("rt_receiver", "sf"), RADIO_SF))
RT_OVERSAMPLE = int(_get(("rt_receiver", "oversample"), RADIO_OVERSAMPLE))
RT_GATEWAY_FREQ_HZ = int(_get(("rt_receiver", "gateway_freq_hz"), RADIO_FREQUENCY_HZ))
RT_PREAMBLE_THRESH = int(_get(("rt_receiver", "preamble_thresh"), LORA_RX_PREAMBLE_THRESH))
RT_USRP_ADDR = str(_get(("rt_receiver", "usrp_addr"), "10.44.0.6"))
RT_SDRPLAY_URL = str(_get(("rt_receiver", "sdrplay_url"), SDRPLAY_URL))
