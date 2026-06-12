# LoRaWAN SDR Scanner — LoRaPHY Backend + Cyber GUI

This project is a clean bachelor-thesis starter implementation for a low-cost RTL-SDR LoRaPHY scanner.
It focuses on LoRa physical-layer work only: channelization, preamble detection, synchronization/SFD, dechirp + FFT symbol demodulation, explicit PHY header debug/PHDR CRC, payload symbol extraction, and JSONL logging.

The GUI is a real scanner controller, not only a log viewer. It can start/stop RTL-SDR live scanning or replay offline IQ files, while displaying a neon dashboard with wideband spectrum, channel spectrum, packet logger table, channel activity, PHDR status, raw header bins, raw payload symbols, CFO/STO, SNR/relative power, and detailed JSONL records.

## Install

```bash
pip install -r requirements.txt
```

For live RTL-SDR mode on Windows, install the RTL-SDR driver with Zadig and make sure `pyrtlsdr` can open the dongle.

## Run the GUI

```bash
python -m lora_scanner.gui
```

or double-click:

```text
run_gui.bat
```

Recommended first real LilyGO test settings:

```text
Source: rtlsdr
RF center Hz: 868.2e6
Channels: 868.1e6
SF: 8
BW Hz: 125e3
Input sample rate: 1.5e6
JSONL log: logs/gui_live_sf8.jsonl
```

8-channel EU868 scan:

```text
Source: rtlsdr
RF center Hz: 868.8e6
Channels: eu868
SF: 8
BW Hz: 125e3
Input sample rate: 2.0e6
```

## Headless commands

```bash
python -m py_compile lora_scanner/*.py
python -m pytest -q
python -m lora_scanner.receiver --source rtlsdr --center 868.2e6 --channels 868.1e6 --sf 8 --bw 125e3 --sample-rate 1.5e6 --jsonl logs/live_sf8.jsonl
python -m lora_scanner.receiver --source file --iq captures/your_capture.npz --center 868.1e6 --channels 868.1e6 --sf 8 --bw 125e3 --sample-rate 125e3 --jsonl logs/replay.jsonl
```

## Important notes

- This is LoRaPHY only, not LoRaWAN MAC/decryption/device identification.
- The GUI is wired to the actual scanner backend.
- Payload CRC failure is logged with raw symbols and decoder debug information so timing/CFO/Gray/interleaving/Hamming/whitening issues can be diagnosed.
- Offline tests are included for sanity; real acceptance is real over-the-air LilyGO packets.

## Exact ZIP Decoder Mode

This cleaned build keeps only the original ZIP decoder pieces that are still
used by the live scanner under the top-level `LoRa_Rx_SDR/` package:

- `LoRa_Rx_SDR/Decode_Symbols.py`
- `LoRa_Rx_SDR/lora_sdr_config.py`

The live scanner/GUI still uses the clean RTL-SDR/channelizer front-end, but after
header symbols are extracted it first calls the exact original
`Decode_Symbols.Get_Header_Info()`. When that passes, it tries the exact original
`Decode_Symbols.Get_Data_Info()` for payload decoding, then falls back to the
clean payload decoder only if the clean header convention matches the same
expected PHDR/profile. This preserves the exact ZIP header path while allowing
validated payload recovery when the copied payload function rejects otherwise
stable raw bins.

In JSONL records, look for:

```json
"legacy_zip_phdr": {...},
"legacy_zip_payload_decode": {...},
"clean_payload_decode": {...},
"phdr": {"source": "exact_zip_Decode_Symbols.py", ...}
```

In the GUI table, a PHDR status starting with `ZIP` means the original ZIP decoder
path passed the explicit-header check.

Offline replay mode still works, but this cleaned copy does not include bundled
`.npz` captures. Put your own IQ capture under `captures/` or point `--iq` at an
existing `.npy`, `.npz`, `.dat`, or `.cf32` IQ file.

## Candidate filtering

The scanner now separates real packets from raw detector candidates:

- Main packet table: PHDR-valid and full decoded packet rows; debug/no-PHDR candidates stay in the Debug candidates tab.
- In expected-payload validation mode, packet classes are separated: `phdr_only_packet` means preamble/sync/explicit header passed but payload CRC did not; `full_payload_packet` means PHDR passed and the payload CRC-backed decode matches the configured ASCII text.
- Main packet table and CSV export now split `Header raw bins`, `Payload raw bins`, `Decode status`, `Decoded ASCII`, and `Payload CRC`, so payload CRC failures no longer hide the extracted LoRa symbol bins.
- The table also exports `Hdr off [samp]` and `Payload off [samp]`, and the selected-packet details show the compact payload timing sweep attempted after valid PHDR.
- Debug candidates tab: NO PHDR, ambiguous PHDR, short windows, and other failed detector windows.
- After an accepted packet, the receiver ignores same-channel candidates for 1.8 seconds by default to reduce duplicate rows from the same transmission.
- The SFD column is labeled as raw sync-bin display and CFO is labeled as an estimate. Treat SFD/CFO as synchronization diagnostics, not calibrated thesis evidence; use PHDR and payload CRC status as packet evidence.

For the current LilyGO payload test, the GUI enables a stricter expected-payload
profile by default. The GUI field is now set to `JANA`, so the expected length is
derived automatically from that text:

- expected exact-ZIP PHDR: `L=4 CR=4/5 CRC=1`
- expected payload text: `JANA`
- all other PHDRs are routed to Debug candidates
- header and payload decode both sweep `-1 symbol`, `+1 symbol`, and fine offsets `-64..+64` samples in 16-sample steps; the receiver selects the header offset that matches the expected PHDR and the payload offset that gives the expected text or CRC OK
- near-miss one-byte-short headers are tried against the configured expected length only as a rescue path; the GUI labels these as `ZIP*` and promotes them only if payload decoding confirms the expected text or CRC OK

Headless equivalent:

```bash
python -m lora_scanner.receiver --source rtlsdr --center 868.2e6 --channels 868.1e6 --sf 8 --bw 125e3 --sample-rate 1.5e6 --jsonl logs/live_sf8.jsonl --expected-payload-ascii JANA
```

## Project architecture, input, and output reference

### End-to-end block diagram

```text
+----------------------------------------------------------------------------+
| 1) INPUT SOURCES                                                            |
|                                                                            |
|    lora_scanner.sdr_source.RtlSdrSource                                     |
|    IN : center_freq, sample_rate, gain, block_size                          |
|    OUT: live RTL-SDR complex64 IQ blocks                                    |
|                                                                            |
|    lora_scanner.iq_file_source.IQFileSource                                 |
|    IN : .npy / .npz / .dat / .cf32 IQ file, block_size                      |
|    OUT: offline replay complex64 IQ blocks                                  |
+----------------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------------+
| 2) CHANNELIZATION                                                           |
|                                                                            |
|    lora_scanner.channelizer.Channelizer                                     |
|    IN : wideband IQ block, RF center Hz, channel list, BW, oversample       |
|    FLOW: remove DC -> mix channel to baseband -> FIR low-pass -> decimate   |
|    OUT: ChannelOutput(center_hz, sample_rate, samples) for each channel     |
+----------------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------------+
| 3) PREAMBLE DETECTION                                                       |
|                                                                            |
|    lora_scanner.detector.PreambleDetector                                   |
|    IN : per-channel baseband samples                                        |
|    FLOW: dechirp symbols -> FFT -> require repeated stable bins             |
|    OUT: PreambleCandidate(sample_index, bins, SNR, isolation, RSSI)         |
+----------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------- decision -----------------------------------+
| Is ReceiverConfig.oversample greater than 1?                                |
+----------------------------------------------------------------------------+
               | yes                                                | no
               v                                                    v
+---------------------------------------+        +--------------------------------+
| 4A) OVERSAMPLED SYNCHRONIZATION       |        | 4B) CLASSIC SYNCHRONIZATION   |
|                                       |        |                                |
| lora_scanner.sync2.Synchronizer2      |        | lora_scanner.synchronizer.    |
| IN : oversampled channel samples,     |        | Synchronizer                  |
|      candidate index                  |        | IN : samples, candidate index,|
| FLOW: lock SFD -> estimate joint      |        |      base sample index        |
|       CFO/STO -> correct CFO ->       |        | FLOW: refine preamble timing, |
|       align header/payload starts     |        |       estimate CFO, find SFD  |
| OUT: Sync2Result with SFD/header/     |        | OUT: SyncResult with preamble,|
|      payload starts, CFO, STO,        |        |      SFD/header starts, CFO,  |
|      corrected stream metadata        |        |      STO, SNR/RSSI            |
+---------------------------------------+        +--------------------------------+
               |                                                    |
               +-------------------------+--------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------+
| 5) SYMBOL DEMODULATION                                                      |
|                                                                            |
|    lora_scanner.demodulator.demod_symbols                                   |
|    IN : synchronized header/payload sample windows                          |
|    FLOW: multiply by chirp reference -> FFT -> select peak bins             |
|    OUT: SymbolMetrics(bins, SNR, isolation, peak_power, top_bins)           |
|                                                                            |
|    Helper file: lora_scanner.chirps.py                                      |
|    IN : SF, BW, sample_rate, CFO, symbol value                              |
|    OUT: upchirps, downchirps, modulated symbols, CFO-corrected samples      |
+----------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------+
| 6) HEADER / PHDR DECODING                                                   |
|                                                                            |
|    Preferred legacy path:                                                   |
|    lora_scanner.exact_zip_decoder.decode_header_exact                       |
|    IN : 8 raw header FFT bins, SF                                           |
|    OUT: ExactZipHeader(ok, payload_len, CR, CRC flag, extra bits)           |
|                                                                            |
|    Clean diagnostic path:                                                   |
|    lora_scanner.phy_decode.decode_header                                    |
|    IN : 8 raw header FFT bins, SF                                           |
|    OUT: HeaderCandidate records with PHDR fields, CRC status, convention   |
+----------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------+
| 7) PAYLOAD DECODING                                                         |
|                                                                            |
|    Legacy ZIP payload decoder:                                              |
|    lora_scanner.exact_zip_decoder.decode_payload_exact                      |
|    IN : payload raw FFT bins, SF, ExactZipHeader                            |
|    OUT: ExactZipPayload(ok, payload_hex, payload_ascii, CRC status)         |
|                                                                            |
|    Clean fallback decoder:                                                  |
|    lora_scanner.phy_decode.decode_payload                                   |
|    IN : payload raw FFT bins, SF, HeaderCandidate                           |
|    OUT: PayloadDecodeResult(payload bytes, ASCII, CRC, Hamming debug)       |
+----------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------+
| 8) RECEIVER RECORD BUILDING                                                 |
|                                                                            |
|    lora_scanner.receiver.LoRaReceiver                                       |
|    IN : ReceiverConfig, JsonlLogger, ChannelOutput samples                  |
|    FLOW: buffer samples -> group nearby candidates -> rank decodes ->       |
|          apply valid-packet lockout -> classify packet/debug candidate      |
|    OUT: packet/candidate dictionary with sync, PHDR, payload, CRC, bins,    |
|         timing offsets, failure reason, and candidate class                 |
+----------------------------------------------------------------------------+
                         |                                   |
                         v                                   v
+---------------------------------------+        +--------------------------------+
| 9A) HEADLESS OUTPUT                  |        | 9B) GUI OUTPUT                 |
|                                       |        |                                |
| lora_scanner.logger.JsonlLogger       |        | lora_scanner.gui.MainWindow   |
| IN : packet/candidate dictionaries    |        | IN : worker signals carrying  |
| OUT: JSONL file lines + stdout        |        |      packets, spectra, stats  |
|                                       |        | OUT: packet table, debug tab, |
| CLI entry:                            |        |      spectra, detail JSON,    |
| python -m lora_scanner.receiver       |        |      CSV export               |
|                                       |        |                                |
| Config object:                        |        | GUI entry:                    |
| lora_scanner.config.ReceiverConfig    |        | python -m lora_scanner.gui    |
+---------------------------------------+        +--------------------------------+
```

### File-level purpose, flow, inputs, and outputs

```text
+----------------------------------------------------------------------------+
| FILE: run_gui.bat                                                           |
| PURPOSE: Windows launcher for the graphical scanner.                        |
| INPUT  : User double-click or terminal execution.                           |
| OUTPUT : Runs "python -m lora_scanner.gui", then pauses the console.        |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: requirements.txt                                                      |
| PURPOSE: Dependency list for the Python environment.                        |
| INPUT  : pip install -r requirements.txt                                    |
| OUTPUT : Installs numerical, GUI, SDR, and decoder packages.                |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: config/lora_sdr.json                                                  |
| PURPOSE: Runtime JSON for the preserved legacy decoder configuration.       |
| INPUT  : Debug and radio settings.                                          |
| OUTPUT : Values consumed by LoRa_Rx_SDR/lora_sdr_config.py.                 |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/__init__.py                                              |
| PURPOSE: Marks lora_scanner as an importable Python package.                |
| INPUT  : Python import system.                                              |
| OUTPUT : No direct runtime output.                                          |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/config.py                                                |
| PURPOSE: Defines EU868 channel defaults and ReceiverConfig.                 |
| FLOW   : Computes derived values such as bin count, symbol size, channel    |
|          sample rate, and maximum packet span.                              |
| INPUT  : CLI or GUI settings such as SF, BW, sample rate, channels,         |
|          payload profile, and timing sweep parameters.                      |
| OUTPUT : A typed receiver configuration object used by backend and GUI.     |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/sdr_source.py                                            |
| PURPOSE: Wraps pyrtlsdr for live SDR capture.                               |
| FLOW   : Delays RTL-SDR import so offline replay does not need drivers.     |
| INPUT  : Center frequency, sample rate, gain, block size.                   |
| OUTPUT : Infinite generator of complex64 IQ sample blocks from RTL-SDR.     |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/iq_file_source.py                                        |
| PURPOSE: Loads offline IQ captures and yields them in blocks.               |
| FLOW   : Accepts complex vectors or real I/Q pairs.                         |
| INPUT  : .npy, .npz, .dat, or .cf32 IQ file path.                           |
| OUTPUT : Complex64 IQ array and block generator.                            |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/channelizer.py                                           |
| PURPOSE: Converts wideband IQ into one baseband stream per LoRa channel.    |
| FLOW   : Remove DC -> mix target channel to baseband -> FIR low-pass ->     |
|          decimate to the channel sample rate.                               |
| INPUT  : Wideband IQ samples, RF center frequency, target channels, BW,     |
|          oversample.                                                        |
| OUTPUT : List of ChannelOutput objects, one per channel.                    |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/chirps.py                                                |
| PURPOSE: Provides LoRa chirp and CFO helper functions.                      |
| FLOW   : Creates upchirps, downchirps, modulated symbols, signed FFT-bin    |
|          values, and CFO-corrected samples.                                 |
| INPUT  : SF, BW, sample rate, symbol value, CFO estimate.                   |
| OUTPUT : Chirp arrays, signed bin numbers, frequency-corrected samples.     |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/demodulator.py                                           |
| PURPOSE: Converts LoRa symbol samples into FFT bins.                        |
| FLOW   : Dechirp symbol -> FFT -> choose peak bin -> measure SNR/isolation. |
| INPUT  : Symbol samples, or a sample buffer plus start/count.               |
| OUTPUT : SymbolMetrics with bins, SNR, isolation, peak power, top-k bins.   |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/detector.py                                              |
| PURPOSE: Detects repeated stable LoRa preamble bins.                        |
| FLOW   : Scan channel samples -> demodulate each symbol -> require stable   |
|          repeated FFT bins above SNR/isolation gates.                       |
| INPUT  : Baseband channel samples and base sample index.                    |
| OUTPUT : PreambleCandidate records with sample index, bins, SNR,            |
|          isolation, RSSI, timestamp.                                        |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/synchronizer.py                                          |
| PURPOSE: Classic 1-sample-per-chip synchronizer.                            |
| FLOW   : Refine preamble timing -> estimate CFO from preamble bins ->       |
|          search SFD downchirps -> predict header start.                     |
| INPUT  : Samples, detector candidate index, base sample index.              |
| OUTPUT : SyncResult with timing positions, CFO, STO, bins, SNR/RSSI,        |
|          and failure reason.                                                |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/sync2.py                                                 |
| PURPOSE: Oversampled synchronizer for real over-the-air signals.            |
| FLOW   : Lock directly to SFD -> separate CFO and STO using up/down chirp   |
|          bins -> correct CFO -> align the symbol grid.                      |
| INPUT  : Oversampled channel samples and detector candidate index.          |
| OUTPUT : Sync2Result with SFD/header/payload starts, CFO/STO, corrected     |
|          stream metadata, bins, SNR/RSSI.                                   |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/phy_decode.py                                            |
| PURPOSE: Clean LoRa PHY header and payload decoder.                         |
| FLOW   : Convert bins to bits -> deinterleave -> Hamming correction ->      |
|          validate PHDR CRC -> dewhiten payload -> check payload CRC.        |
| INPUT  : Raw header or payload bins, SF, selected header convention.        |
| OUTPUT : HeaderCandidate or PayloadDecodeResult with decoded fields,        |
|          payload bytes, CRC status, and debug metadata.                     |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/exact_zip_decoder.py                                     |
| PURPOSE: Adapter around the copied original LoRa_Rx_SDR Decode_Symbols.py.  |
| FLOW   : Calls the preserved legacy decoder for PHDR and payload decoding.  |
| INPUT  : Raw header/payload FFT bins and SF.                                |
| OUTPUT : ExactZipHeader and ExactZipPayload records with PHDR fields,       |
|          payload hex/ascii, CRC status, and failure reason.                 |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/logger.py                                                |
| PURPOSE: JSONL logger for packet/candidate records.                         |
| INPUT  : Python dictionaries generated by the receiver.                     |
| OUTPUT : One JSON object per line written to file, plus stdout echo.        |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/receiver.py                                              |
| PURPOSE: Headless backend and CLI entrypoint.                               |
| FLOW   : Build source/channelizer -> buffer per-channel samples -> detect   |
|          candidates -> synchronize -> decode header/payload -> rank nearby  |
|          candidates -> apply lockout -> write accepted/debug records.       |
| INPUT  : CLI arguments or ReceiverConfig; IQ blocks from file or RTL-SDR.   |
| OUTPUT : JSONL packet/candidate records containing channel, sync, PHDR,     |
|          payload, CRC, raw bins, and failure/debug details.                 |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: lora_scanner/gui.py                                                   |
| PURPOSE: PyQt scanner controller and dashboard.                             |
| FLOW   : Read GUI settings -> start worker thread -> run backend -> show    |
|          spectra, channel stats, packets, debug candidates, details.        |
| INPUT  : User GUI controls, optional IQ file, live RTL-SDR stream.          |
| OUTPUT : GUI tables, spectrum plots, detailed JSON view, JSONL log,         |
|          optional CSV file.                                                 |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: LoRa_Rx_SDR/__init__.py                                               |
| PURPOSE: Marks the preserved legacy decoder folder as an importable package.|
| INPUT  : Python import system.                                              |
| OUTPUT : No direct runtime output.                                          |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: LoRa_Rx_SDR/lora_sdr_config.py                                        |
| PURPOSE: Loads legacy configuration constants.                              |
| INPUT  : config/lora_sdr.json and optional LORA_SDR_CONFIG_PATH.            |
| OUTPUT : Module constants for debug flags, radio settings, and legacy       |
|          RX/TX defaults.                                                    |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| FILE: LoRa_Rx_SDR/Decode_Symbols.py                                         |
| PURPOSE: Original LoRa symbol decoder preserved from the ZIP.               |
| FLOW   : Build Hamming/CRC/whitening tables -> decode PHDR -> decode        |
|          payload bits -> check CRC -> store last decoded bytes.             |
| INPUT  : Header symbols, payload symbols, SF, CR, payload length, CRC flag, |
|          extra header bits.                                                 |
| OUTPUT : Header tuple, payload CRC boolean, and decoded PHY/MAC bytes via   |
|          Get_Decoded_Bytes().                                               |
+----------------------------------------------------------------------------+
```

### Main classes and data objects

```text
+----------------------------------------------------------------------------+
| OBJECT: ReceiverConfig                                                      |
| FILE  : lora_scanner/config.py                                              |
| INPUT : Scanner settings.                                                   |
| OUTPUT: Shared configuration plus derived symbol/channel sizes.             |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: ChannelOutput                                                       |
| FILE  : lora_scanner/channelizer.py                                         |
| INPUT : Channel center, output sample rate, samples.                        |
| OUTPUT: Container passed from channelizer to receiver.                      |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: SymbolMetrics                                                       |
| FILE  : lora_scanner/demodulator.py                                         |
| INPUT : Demodulated FFT result lists.                                       |
| OUTPUT: Bins, SNR, isolation, peak power, and top-k candidates.             |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: PreambleCandidate                                                   |
| FILE  : lora_scanner/detector.py                                            |
| INPUT : Stable repeated preamble-bin detection.                             |
| OUTPUT: Candidate packet start metadata for synchronization.                |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: PreambleDetector                                                    |
| FILE  : lora_scanner/detector.py                                            |
| INPUT : Channel samples.                                                    |
| OUTPUT: List of PreambleCandidate objects.                                  |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: SyncResult / Synchronizer                                           |
| FILE  : lora_scanner/synchronizer.py                                        |
| INPUT : Candidate timing and channel samples.                               |
| OUTPUT: Classic sync result for the non-oversampled path.                   |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: Sync2Result / Synchronizer2                                         |
| FILE  : lora_scanner/sync2.py                                               |
| INPUT : Oversampled channel samples and candidate timing.                   |
| OUTPUT: SFD-locked sync result with joint CFO/STO and aligned starts.       |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: HeaderCandidate                                                     |
| FILE  : lora_scanner/phy_decode.py                                          |
| INPUT : Header bins decoded under one convention.                           |
| OUTPUT: PHDR fields, CRC status, convention, Hamming debug data.            |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: PayloadDecodeResult                                                 |
| FILE  : lora_scanner/phy_decode.py                                          |
| INPUT : Payload bins plus selected header.                                  |
| OUTPUT: Payload bytes/ASCII/hex, CRC status, Hamming errors, failure reason.|
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: ExactZipHeader / ExactZipPayload                                    |
| FILE  : lora_scanner/exact_zip_decoder.py                                   |
| INPUT : Raw bins passed to the legacy decoder.                              |
| OUTPUT: Legacy-compatible PHDR and payload decode records.                  |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: IQFileSource                                                        |
| FILE  : lora_scanner/iq_file_source.py                                      |
| INPUT : Offline IQ file path and block size.                                |
| OUTPUT: Generator of IQ sample blocks.                                      |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: RtlSdrSource                                                        |
| FILE  : lora_scanner/sdr_source.py                                          |
| INPUT : RTL-SDR frequency, sample rate, gain, block size.                   |
| OUTPUT: Generator of live IQ sample blocks.                                 |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: JsonlLogger                                                         |
| FILE  : lora_scanner/logger.py                                              |
| INPUT : Record dictionaries.                                                |
| OUTPUT: JSONL file lines and stdout lines.                                  |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: LoRaReceiver                                                        |
| FILE  : lora_scanner/receiver.py                                            |
| INPUT : ReceiverConfig, JsonlLogger, channel samples.                       |
| OUTPUT: Packet/candidate JSON records and packet counts.                    |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: GuiRunConfig                                                        |
| FILE  : lora_scanner/gui.py                                                 |
| INPUT : Values read from GUI controls.                                      |
| OUTPUT: GUI-specific run settings converted into ReceiverConfig.            |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: ScannerWorker                                                       |
| FILE  : lora_scanner/gui.py                                                 |
| INPUT : GuiRunConfig.                                                       |
| OUTPUT: Qt signals carrying packets, spectra, channel stats, status, errors.|
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: SpectrumWidget                                                      |
| FILE  : lora_scanner/gui.py                                                 |
| INPUT : Frequency and power arrays.                                         |
| OUTPUT: Painted spectrum plot in the GUI.                                   |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: EmittingJsonlLogger                                                 |
| FILE  : lora_scanner/gui.py                                                 |
| INPUT : Record dictionaries from backend.                                   |
| OUTPUT: Writes JSONL and emits Qt packet/console signals.                   |
+----------------------------------------------------------------------------+

+----------------------------------------------------------------------------+
| OBJECT: MainWindow                                                          |
| FILE  : lora_scanner/gui.py                                                 |
| INPUT : User interaction and worker signals.                                |
| OUTPUT: Scanner dashboard, packet/debug tables, details panel, CSV export.  |
+----------------------------------------------------------------------------+
```
