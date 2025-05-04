# Software-Defined Radio (SDR) Application Suite

Advanced Signal Processing and Machine Learning for Cognitive Radio Systems

## Overview

This repository contains the source code and documentation for the **Software-Defined Radio (SDR) Application Suite**, a research project focused on real-time signal acquisition, processing, and protocol analysis using advanced digital signal processing and machine learning techniques.

The suite integrates deep learning-driven signal classification, multi-protocol decoding, and a modular architecture to support diverse wireless communication domains including aviation, maritime, IoT, and satellite systems.

## Features

- **Cognitive Signal Processing Framework**  
  Real-time identification of 16+ modulation types (AM, FM, SSB, PSK, QAM, FSK, LoRa, etc.) with adaptive parameter selection and anomaly detection.

- **Multi-Protocol Decoder Engine**  
  Unified decoding for aviation (ADS-B, ACARS), maritime (AIS, NAVTEX), satellite (NOAA APT), IoT (LoRa, Sigfox), digital voice (DMR, D-STAR), and paging protocols.

- **Cross-Platform Visualization**  
  Qt-based GUI with multi-perspective spectrum visualization, protocol-specific views, geographic mapping, and time-series signal strength analysis.

- **Machine Learning Engine**  
  TensorFlow-based convolutional neural network for spectral pattern recognition with 27-class classification and optimized inference latency (~15 ms).

- **Modular Architecture**  
  Hardware abstraction layer supporting multiple SDR devices (RTL-SDR, HackRF, Airspy, USRP), plugin-based protocol decoders, and extensible framework for community contributions.

## System Architecture

The suite employs a modular four-layer design:

1. **Hardware Abstraction Layer** - Device-agnostic control via SoapySDR API with dynamic parameter management and hot-swapping.
2. **Signal Processing Core** - Advanced FFT processing, adaptive demodulation, and spectral refinement.
3. **Machine Learning Engine** - Deep learning models optimized for real-time modulation classification.
4. **Protocol Decoding Framework** - Plugin architecture with ML-driven protocol identification and adaptive resource allocation.

## Installation

*Instructions to install dependencies and set up the environment will be added here.*

## Usage

*Basic usage instructions, command-line options, and GUI launch commands will be provided here.*

## Performance Highlights

- Real-time processing latency under 20 ms.
- Top-1 modulation classification accuracy of 91.7%.
- Robust decoding across multiple protocols and dynamic RF environments.
- Adaptive spectrum learning with minimal user intervention.

## Research Contributions

- Novel integration of cognitive radio principles with accessible SDR hardware.
- Deep learning approach for real-time modulation classification.
- Unified cross-domain protocol decoding methodology.
- Framework enabling community-driven protocol extension.

## Limitations

- Maximum real-time bandwidth processing limited to ~10 MHz on consumer hardware.
- Performance depends on hardware capabilities and training dataset coverage.
- Requires minimum quad-core CPU and 8GB RAM for reliable operation.

## Future Work

- Expand protocol decoder library and community SDK.
- Optimize processing bandwidth with hardware acceleration.
- Enhance model generalization for emerging signal types.

## Citation

If you use this software or research in your work, please cite:

*Author(s), "Software-Defined Radio (SDR) Application Suite: Advanced Signal Processing and Machine Learning for Cognitive Radio Systems," [Conference/Journal], Year.*

## License

*Specify the license under which this project is released.*

## Contact

For questions or contributions, please contact:  
Sankur Kundu – heyobsidian93@gmail.com

---

This project aims to democratize advanced spectrum monitoring capabilities by combining state-of-the-art DSP and AI techniques with accessible hardware platforms.
