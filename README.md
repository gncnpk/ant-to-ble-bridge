# ANT+ to BLE FTMS Bridge

This application bridges ANT+ devices (heart rate monitors and power meters) to Bluetooth Low Energy (BLE) clients using the Fitness Machine Service (FTMS) standard.

## Features

- Receives data from ANT+ heart rate monitors and power meters
- Exposes data via BLE FTMS (Fitness Machine Service)
- Supports power, cadence, speed, and heart rate data
- Compatible with cycling apps like Zwift, TrainerRoad, etc.
- Cross-platform support (Windows, Linux, macOS)

## Requirements

- Python 3.8+
- ANT+ USB stick
- ANT+ devices (heart rate monitor, power meter/smart trainer)
- Administrator/sudo privileges (for BLE and ANT+ access)