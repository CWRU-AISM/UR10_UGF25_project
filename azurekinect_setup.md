# Azure Kinect SDK Setup on Ubuntu 24.04

This documents the setup procedure for the Azure Kinect SDK on Ubuntu 24.04.3 LTS.

## Background

Microsoft officially supports the Azure Kinect SDK only up to Ubuntu 18.04. However, the SDK can be installed on newer Ubuntu versions using one of two approaches:

1. **Pre-built .deb packages** (used here) - Download and install the Ubuntu 18.04 packages directly
2. **Build from source** - Use the UM-ARM-Lab fork which patches the SDK for Ubuntu 22+

## References

This setup followed the **jlblancoc gist method** (pre-built packages):

- [jlblancoc's gist for Ubuntu 22.04+](https://gist.github.com/jlblancoc/ae2a082b0ed5af2e71645b04b7207210) - **Primary guide used**
- [UM-ARM-Lab Azure Kinect SDK Ubuntu 22 Fork](https://github.com/UM-ARM-Lab/Azure-Kinect-Sensor-SDK-Ubuntu-22) - Alternative: build from source
  - [Usage docs](https://github.com/UM-ARM-Lab/Azure-Kinect-Sensor-SDK-Ubuntu-22/blob/develop/docs/usage.md)
- [Official Microsoft 99-k4a.rules](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/scripts/99-k4a.rules) - udev rules source

## Installation Steps

### 1. Download the .deb packages

Download the packages from Microsoft's Ubuntu 18.04 repository:

```bash
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb
```

### 2. Install missing dependency (libsoundio1)

**Important:** `k4a-tools` requires `libsoundio1` which is not available in Ubuntu 24.04. Download and install it from Ubuntu 20.04 repositories:

```bash
wget http://archive.ubuntu.com/ubuntu/pool/universe/libs/libsoundio/libsoundio1_1.1.0-1_amd64.deb
sudo apt install ./libsoundio1_1.1.0-1_amd64.deb
```

### 3. Install the Azure Kinect packages

```bash
sudo apt install ./libk4a1.4_1.4.1_amd64.deb
sudo apt install ./libk4a1.4-dev_1.4.1_amd64.deb
sudo apt install ./k4a-tools_1.4.1_amd64.deb
```

### 4. Configure udev rules for non-root access

Create the udev rules file to allow accessing the camera without root privileges:

```bash
sudo tee /etc/udev/rules.d/99-k4a.rules << 'EOF'
# Bus 002 Device 116: ID 045e:097a Microsoft Corp.  - Generic Superspeed USB Hub
# Bus 001 Device 015: ID 045e:097b Microsoft Corp.  - Generic USB Hub
# Bus 002 Device 118: ID 045e:097c Microsoft Corp.  - Azure Kinect Depth Camera
# Bus 002 Device 117: ID 045e:097d Microsoft Corp.  - Azure Kinect 4K Camera
# Bus 001 Device 016: ID 045e:097e Microsoft Corp.  - Azure Kinect Microphone Array

BUS!="usb", ACTION!="add", SUBSYSTEM!=="usb_device", GOTO="k4a_logic_rules_end"

ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097a", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097b", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097c", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097d", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097e", MODE="0666", GROUP="plugdev"

LABEL="k4a_logic_rules_end"
EOF
```

### 5. Reload udev rules

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 6. Reconnect the Azure Kinect device

Unplug and replug the Azure Kinect for the new udev rules to take effect.

## Installed Components

### Packages
| Package | Version | Description |
|---------|---------|-------------|
| libk4a1.4 | 1.4.1 | Runtime libraries |
| libk4a1.4-dev | 1.4.1 | Development headers and CMake files |
| k4a-tools | 1.4.1 | k4aviewer, k4arecorder, AzureKinectFirmwareTool |

### File Locations

**Libraries:**
- `/usr/lib/x86_64-linux-gnu/libk4a.so.1.4.1`
- `/usr/lib/x86_64-linux-gnu/libk4arecord.so.1.4.1`
- `/usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0` (closed-source depth engine)

**Headers:**
- `/usr/include/k4a/` (k4a.h, k4a.hpp, k4atypes.h, etc.)
- `/usr/include/k4arecord/` (playback.h, record.h, etc.)

**CMake files:**
- `/usr/lib/x86_64-linux-gnu/cmake/k4a/`
- `/usr/lib/x86_64-linux-gnu/cmake/k4arecord/`

**Tools:**
- `/usr/bin/k4aviewer` - GUI viewer for camera streams
- `/usr/bin/k4arecorder` - Record to MKV files
- `/usr/bin/AzureKinectFirmwareTool` - Firmware update utility

**udev rules:**
- `/etc/udev/rules.d/99-k4a.rules`

## Usage

### Test the camera
```bash
k4aviewer
```

### Record video
```bash
k4arecorder -d WFOV_2X2BINNED -c 1080p -r 30 -l 10 output.mkv
```

### CMake integration
```cmake
find_package(k4a REQUIRED)
target_link_libraries(your_target k4a::k4a)
```

## Troubleshooting

### Permission denied when accessing camera
- Ensure you're in the `plugdev` group: `groups $USER`
- Add yourself if needed: `sudo usermod -aG plugdev $USER` (then log out and back in)
- Verify udev rules are loaded: `udevadm info -a -n /dev/bus/usb/XXX/YYY | grep idProduct`

### Depth engine not found
The depth engine (`libdepthengine.so.2.0`) must be in the library path. It should be automatically found at `/usr/lib/x86_64-linux-gnu/libk4a1.4/`.

## System Info

- **OS:** Ubuntu 24.04.3 LTS (Noble Numbat)
- **Kernel:** 6.8.0-88-lowlatency
- **SDK Version:** 1.4.1
