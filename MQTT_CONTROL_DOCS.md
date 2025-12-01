# SparkNode MQTT Control Script - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Architecture](#architecture)
8. [Command Reference](#command-reference)

---

## Overview

The **SparkNode MQTT Control Script** is a Python application that provides complete control over SparkNode robots via MQTT. It acts as a client interface to send commands to ESP32-based robots running the SparkNode firmware.

### Key Features
- **Configurable parameters** at the top of the file (no code changes needed)
- **Automatic timing** - handles sleep calculations automatically
- **Pre-built sequences** - ready-to-use movement patterns
- **Interactive mode** - send custom commands on demand
- **Real-time feedback** - receives status updates from robots
- **Error handling** - graceful disconnection and Ctrl+C handling
- **Verbose output** - detailed logging of all actions

---

## Installation

### Prerequisites
- Python 3.7 or higher
- `pip` (Python package manager)
- MQTT broker running (Mosquitto on Raspberry Pi)
- Network connectivity to the broker

### Step 1: Install Required Package

```bash
pip install paho-mqtt
```

### Step 2: Download the Script

Save `sparknode_mqtt_control.py` to your machine.

### Step 3: Make it Executable (Optional)

```bash
chmod +x sparknode_mqtt_control.py
```

### Step 4: Verify Setup

Test MQTT connection to your broker:
```bash
mosquitto_pub -h <your-pi-ip> -t "test" -m "hello"
```

---

## Configuration

All configuration is done at the **top of the file** (lines 11-61). No code changes needed elsewhere.

### MQTT Broker Settings

```python
MQTT_BROKER = "192.168.11.100"  # IP address of your Raspberry Pi
MQTT_BROKER_PORT = 1883          # Default MQTT port (rarely changes)
MQTT_KEEPALIVE = 60              # Keep-alive timeout in seconds
```

**Finding your Broker IP:**
```bash
# On your Raspberry Pi
hostname -I
```

### Target SparkNode

```python
SPARKNODE_ID = "sparknode07"  # Which robot to control (sparknode01, sparknode02, etc)
```

The script automatically builds the MQTT topic:
- Command topic: `arena/sparknode07/cmd`
- Status topic: `arena/sparknode07/status`

### Movement Speed Parameters (0-63 scale)

```python
DEFAULT_SPEED = 32        # Default speed (~51% PWM duty cycle)
TURN_SPEED = 30           # Speed used for turning
DRIVE_SPEED = 32          # Speed used for forward/reverse
SLOW_SPEED = 15           # Slow speed for precision movements
FAST_SPEED = 45           # Fast speed for quick movements
```

**Speed Scale Reference:**
- `0` = stopped
- `32` = ~51% power (good default)
- `63` = maximum power

### Duration Parameters (milliseconds)

```python
TURN_DURATION = 1000             # Default turn duration (1 second)
DRIVE_FORWARD_DURATION = 3500    # Default forward duration (3.5 seconds)
DRIVE_REVERSE_DURATION = 2000    # Default reverse duration (2 seconds)
SHORT_WAIT = 2000                # Short wait between commands
```

### Calibration Settings (0.5-2.0)

```python
CALIBRATION_LEFT = 1.0   # Left wheel speed multiplier
CALIBRATION_RIGHT = 1.0  # Right wheel speed multiplier
```

Use calibration to compensate for motor differences:
- `1.0` = no adjustment
- `> 1.0` = motor is too slow, boost it
- `< 1.0` = motor is too fast, slow it down

Example: Robot pulls right?
```python
CALIBRATION_LEFT = 1.05   # Boost left motor
CALIBRATION_RIGHT = 1.0   # Keep right normal
```

### Kick Parameters (16-63 speed, 50-500ms duration)

```python
DRIVE_KICK_SPEED = 25       # Initial torque for driving
DRIVE_KICK_DURATION = 100   # How long to apply kick (ms)
TURN_KICK_SPEED = 40        # Initial torque for turning
TURN_KICK_DURATION = 250    # How long to apply kick (ms)
```

The "kick" is an initial power pulse to overcome static friction. Useful for:
- Breaking in motors
- Starting from dead stop
- Improving reliability

### Rotation Parameters

```python
ROTATION_SPEED = 25        # Speed for magnetometer-based rotation (0-63)
ROTATION_STEP_MS = 20      # Duration of each rotation step (ms)
ROTATION_DELAY_MS = 100    # Delay between heading checks (ms)
ROTATION_TOLERANCE = 10.0  # Acceptable heading error in degrees
```

These control the precision of heading-based rotation using the magnetometer.

### Sensor Parameters (50-10000ms)

```python
SENSOR_DELAY_MS = 500  # Delay between sensor readings
```

Sampling rates:
- `100` = 10 Hz (fast)
- `500` = 2 Hz (normal)
- `1000` = 1 Hz (slow)

### Script Behavior

```python
ENABLE_VERBOSE = True     # Print detailed output (set False to quiet)
BREAK_IN_CYCLES = 5       # Number of cycles for break-in sequences
SAFETY_BUFFER = 1.0       # Extra wait time after commands (seconds)
```

---

## API Reference

### Connection Functions

#### `connect_broker() -> mqtt.Client`

Establishes connection to MQTT broker and returns client object.

```python
client = connect_broker()
```

**Returns:** MQTT client object (already connected)

**Raises:** `SystemExit` if connection fails

---

### Helper Functions

#### `send_command(client: mqtt.Client, command: str) -> None`

Sends raw MQTT command to the robot.

```python
send_command(client, "drive forward 32 5000")
send_command(client, "config set calibration 1.05 1.0")
```

**Parameters:**
- `client`: MQTT client object
- `command`: Raw MQTT command string

**Notes:**
- Automatically prints command if `ENABLE_VERBOSE = True`
- Does not wait for completion
- Used internally by other functions

#### `calculate_sleep_time(duration_ms: int, multiplier: float = SAFETY_BUFFER) -> float`

Calculates proper wait time based on command duration.

```python
sleep_time = calculate_sleep_time(3500)  # Returns 4.5 (3.5 + 1.0)
sleep_time = calculate_sleep_time(1000, multiplier=2.0)  # Returns 3.0
```

**Returns:** Sleep time in seconds

---

### Movement Functions

#### `drive_forward(client, speed=None, duration=None) -> None`

Drive forward at specified speed for duration.

```python
drive_forward(client)  # Uses DRIVE_SPEED and DRIVE_FORWARD_DURATION
drive_forward(client, speed=40, duration=5000)  # Custom: 40 speed, 5 seconds
```

**Parameters:**
- `client`: MQTT client
- `speed`: Optional, 0-63 (default: DRIVE_SPEED)
- `duration`: Optional, milliseconds (default: DRIVE_FORWARD_DURATION)

**Behavior:**
- Sends command
- Automatically waits for completion
- Prints status if verbose

#### `drive_reverse(client, speed=None, duration=None) -> None`

Drive backward at specified speed for duration.

```python
drive_reverse(client, speed=20, duration=2000)
```

**Parameters:** Same as `drive_forward`

#### `turn_left(client, speed=None, duration=None) -> None`

Turn left at specified speed for duration.

```python
turn_left(client)  # Uses defaults
turn_left(client, speed=25, duration=500)
```

**Parameters:** Same as `drive_forward`

#### `turn_right(client, speed=None, duration=None) -> None`

Turn right at specified speed for duration.

```python
turn_right(client, speed=30)
```

**Parameters:** Same as `drive_forward`

#### `rotate_to(client, angle, speed=None, step_ms=None, delay_ms=None, tolerance=None) -> None`

Rotate to specific heading using magnetometer.

```python
rotate_to(client, 90)  # Rotate to East (90 degrees)
rotate_to(client, 0, speed=25, tolerance=3.0)  # Rotate to North with high precision
```

**Parameters:**
- `client`: MQTT client
- `angle`: Target heading in degrees (0-360)
  - `0` = North
  - `90` = East
  - `180` = South
  - `270` = West
- `speed`: Optional, 0-63 (default: ROTATION_SPEED)
- `step_ms`: Optional, rotation step duration (default: ROTATION_STEP_MS)
- `delay_ms`: Optional, delay between checks (default: ROTATION_DELAY_MS)
- `tolerance`: Optional, heading error tolerance (default: ROTATION_TOLERANCE)

**Notes:**
- Waits ~15 seconds for rotation to complete
- Requires magnetometer calibration
- Uses closed-loop heading feedback

#### `stop(client) -> None`

Stop all motors immediately.

```python
stop(client)
```

**Behavior:**
- Applies brake briefly
- Then coast to stop
- Waits 0.5 seconds

---

### Configuration Functions

#### `set_calibration(client, left=None, right=None) -> None`

Set wheel speed calibration factors.

```python
set_calibration(client, left=1.05, right=1.0)
```

**Parameters:**
- `left`: Left motor multiplier (0.5-2.0, default: CALIBRATION_LEFT)
- `right`: Right motor multiplier (0.5-2.0, default: CALIBRATION_RIGHT)

#### `set_drive_kick(client, speed=None, duration=None) -> None`

Configure drive kick-start parameters.

```python
set_drive_kick(client, speed=25, duration=100)
```

**Parameters:**
- `speed`: Kick speed (16-63, default: DRIVE_KICK_SPEED)
- `duration`: Kick duration (50-500ms, default: DRIVE_KICK_DURATION)

#### `set_turn_kick(client, speed=None, duration=None) -> None`

Configure turn kick-start parameters.

```python
set_turn_kick(client, speed=40, duration=250)
```

**Parameters:**
- `speed`: Kick speed (16-63, default: TURN_KICK_SPEED)
- `duration`: Kick duration (50-500ms, default: TURN_KICK_DURATION)

#### `set_default_speed(client, speed) -> None`

Set default movement speed.

```python
set_default_speed(client, 28)
```

**Parameters:**
- `speed`: Default speed (0-63)

#### `show_config(client) -> None`

Display current robot configuration.

```python
show_config(client)
```

**Output:** Prints calibration, kick, and speed settings

---

### Sensor Functions

#### `start_sensor_loop(client, mode="infinite", iterations=None) -> None`

Start IMU sensor data collection.

```python
start_sensor_loop(client)  # Continuous mode
start_sensor_loop(client, mode="counted", iterations=50)  # 50 samples
```

**Parameters:**
- `mode`: "infinite", "stop", or "counted"
- `iterations`: Number of samples (only with mode="counted")

#### `stop_sensor_loop(client) -> None`

Stop sensor data collection.

```python
stop_sensor_loop(client)
```

#### `set_sensor_delay(client, delay_ms=None) -> None`

Set sensor sampling rate.

```python
set_sensor_delay(client, 500)  # 2 Hz sampling
set_sensor_delay(client, 100)  # 10 Hz sampling
```

**Parameters:**
- `delay_ms`: Delay between readings (50-10000ms, default: SENSOR_DELAY_MS)

---

### Calibration Functions

#### `calibrate_gyro(client) -> None`

Recalibrate gyroscope (IMU rotation sensor).

```python
calibrate_gyro(client)
```

**Prerequisites:** Robot must be stationary

**Duration:** ~5 seconds

#### `calibrate_mag(client) -> None`

Recalibrate magnetometer (IMU heading sensor).

```python
calibrate_mag(client)
```

**Prerequisites:** Robot will rotate automatically during calibration

**Duration:** ~10 seconds

**Notes:** Keep hands away during calibration

---

### Sequence Functions

#### `square_pattern_right(client, cycles=None) -> None`

Execute break-in cycle turning right.

```python
square_pattern_right(client)  # Default BREAK_IN_CYCLES
square_pattern_right(client, cycles=10)  # Custom 10 cycles
```

**What it does:**
1. Set initial calibration (1.0, 1.05)
2. For each cycle:
   - Turn right
   - Adjust calibration for forward drive
   - Drive forward
   - Repeat

**Handles:** Keyboard interrupt (Ctrl+C) safely

#### `square_pattern_left(client, cycles=None) -> None`

Execute break-in cycle turning left.

```python
square_pattern_left(client, cycles=5)
```

**Identical to `square_pattern_right` but turns left**

#### `rotation_sequence(client) -> None`

Execute cardinal direction rotation sequence.

```python
rotation_sequence(client)
```

**Sequence:**
1. Start sensor loop (for heading feedback)
2. Rotate to North (0°)
3. Rotate to East (90°)
4. Rotate to South (180°)
5. Rotate to West (270°)
6. Return to North (0°)
7. Stop sensor loop

**Duration:** ~75 seconds total

**Handles:** Keyboard interrupt safely

---

## Usage Examples

### Example 1: Simple Forward Movement

```python
from sparknode_mqtt_control import *

client = connect_broker()

# Drive forward using defaults
drive_forward(client)

# Or with custom parameters
drive_forward(client, speed=40, duration=2000)

client.disconnect()
```

### Example 2: Square Pattern Break-in

```python
client = connect_broker()

# Run 5 cycles of square pattern (default)
square_pattern_right(client)

# Or custom cycles
square_pattern_left(client, cycles=10)

client.disconnect()
```

### Example 3: Rotation Test

```python
client = connect_broker()

# Rotate through cardinal directions
rotation_sequence(client)

client.disconnect()
```

### Example 4: Motor Calibration

```python
client = connect_broker()

# Check current configuration
show_config(client)

# Adjust calibration (robot pulls right - boost left motor)
set_calibration(client, left=1.1, right=1.0)

# Test forward movement
drive_forward(client, speed=32, duration=3000)

client.disconnect()
```

### Example 5: Sensor Collection

```python
client = connect_broker()

# Start continuous sensor collection
start_sensor_loop(client, mode="infinite")

# Let it run for 30 seconds
import time
time.sleep(30)

# Stop collection
stop_sensor_loop(client)

client.disconnect()
```

### Example 6: Custom Sequence

```python
client = connect_broker()

# Create a custom movement pattern
print("Starting custom sequence...")

# Move in a triangle
drive_forward(client, speed=30, duration=2000)
turn_right(client, speed=30, duration=500)

drive_forward(client, speed=30, duration=2000)
turn_right(client, speed=30, duration=500)

drive_forward(client, speed=30, duration=2000)

print("Sequence complete!")
client.disconnect()
```

### Example 7: Interactive Menu (Default)

```bash
python3 sparknode_mqtt_control.py
```

Then:
```
╔══════════════════════════════════════════════════════════╗
║        SparkNode MQTT Control Script (Python)            ║
╚══════════════════════════════════════════════════════════╝

Configuration:
  Broker: 192.168.11.100:1883
  Target: sparknode07
  Default Speed: 32
  Calibration: L=1.0, R=1.0
  
Options:
  1. Square Pattern (Right)
  2. Square Pattern (Left)
  3. Cardinal Rotation Sequence
  4. Custom Commands (Interactive)
  5. Exit

Select option (1-5): 1
```

---

## Troubleshooting

### Connection Timeout

**Error:** `✗ Connection error: timed out`

**Solutions:**
1. Verify Raspberry Pi IP address:
   ```bash
   # On Pi
   hostname -I
   ```

2. Check mosquitto is running:
   ```bash
   # On Pi
   sudo systemctl status mosquitto
   ```

3. Update MQTT_BROKER in script:
   ```python
   MQTT_BROKER = "192.168.1.50"  # Use correct IP
   ```

### DeprecationWarning

**Warning:** `Callback API version 1 is deprecated`

**Fix:** Update line 101:
```python
# Old
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

# New
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
```

### Robot Not Responding

**Check:**
1. Verify SPARKNODE_ID is correct:
   ```python
   SPARKNODE_ID = "sparknode07"  # Check this matches your robot
   ```

2. Test with mosquitto_pub:
   ```bash
   mosquitto_pub -h 192.168.11.100 -t "arena/sparknode07/cmd" -m "stop"
   ```

3. Check robot status messages in output

### Movement Not Working

**Check:**
1. Motor I2C connections on robot
2. Battery voltage
3. Run calibration:
   ```python
   calibrate_gyro(client)
   calibrate_mag(client)
   ```

### Commands Sent But No Response

**Check:**
1. Is `ENABLE_VERBOSE = True`?
2. Check if status messages appear in output
3. Verify robot subscribed to correct topic:
   ```bash
   mosquitto_sub -h 192.168.11.100 -t "arena/sparknode07/status"
   ```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────┐
│  Python Client (your script)            │
│  - Configurable parameters              │
│  - Movement functions                   │
│  - Config functions                     │
│  - Pre-built sequences                  │
└────────────┬────────────────────────────┘
             │
             │ MQTT Commands (TCP)
             │ "drive forward 32 5000"
             │ "config set calibration 1.0 1.05"
             │
             ▼
┌─────────────────────────────────────────┐
│  MQTT Broker (Raspberry Pi)             │
│  - Receives commands                    │
│  - Distributes to robots                │
│  - Collects status messages             │
└────────────┬────────────────────────────┘
             │
             │ MQTT Subscriptions
             │
             ▼
┌─────────────────────────────────────────┐
│  SparkNode Robot (ESP32)                │
│  - mqtt_utils.c (command parser)        │
│  - sparknode_mqtt.c (firmware)          │
│  - FreeRTOS (task scheduler)            │
└────────────┬────────────────────────────┘
             │
             │ I2C Communication
             │
             ▼
┌─────────────────────────────────────────┐
│  Hardware                               │
│  - Motors (DRV8830 driver)              │
│  - IMU (gyro + magnetometer)            │
│  - RGB LED ring                         │
└─────────────────────────────────────────┘
```

### Data Flow

1. **User calls function:** `drive_forward(client, speed=32, duration=5000)`
2. **Function builds command:** `"drive forward 32 5000"`
3. **Command sent via MQTT:** Published to `arena/sparknode07/cmd`
4. **Broker receives:** Routes to subscribed SparkNode
5. **ESP32 receives:** Parses command in mqtt_utils.c
6. **Task spawned:** FreeRTOS task drives motors
7. **Status published:** Robot sends `"driving forward speed=32 5000ms"`
8. **Python receives:** Status callback prints message
9. **Script waits:** Sleeps calculated time (5 + 1 = 6 seconds)
10. **Next command:** Ready for next instruction

### Message Format

MQTT Command Structure:
```
Topic: arena/sparknode07/cmd
Payload: drive forward 32 5000
         ├─ Command: drive
         ├─ Direction: forward
         ├─ Speed: 32 (0-63)
         └─ Duration: 5000 (milliseconds)
```

MQTT Status Structure:
```
Topic: arena/sparknode07/status
Payload: driving forward speed=32 5000ms
```

---

## Command Reference

### Movement Commands

| Command | Parameters | Example |
|---------|-----------|---------|
| `drive forward` | speed (0-63), duration (ms) | `drive forward 32 5000` |
| `drive reverse` | speed (0-63), duration (ms) | `drive reverse 20 2000` |
| `turn left` | speed (0-63), duration (ms) | `turn left 30 1000` |
| `turn right` | speed (0-63), duration (ms) | `turn right 30 1000` |
| `rotate to` | angle (0-360), [speed], [step_ms], [delay_ms], [tolerance] | `rotate to 90 25 20 100 10.0` |
| `stop` | none | `stop` |

### Configuration Commands

| Command | Parameters | Example |
|---------|-----------|---------|
| `config show` | none | `config show` |
| `config reset` | none | `config reset` |
| `config set calibration` | left (0.5-2.0), right (0.5-2.0) | `config set calibration 1.05 1.0` |
| `config set drive_kick` | speed (16-63), duration (50-500) | `config set drive_kick 25 100` |
| `config set turn_kick` | speed (16-63), duration (50-500) | `config set turn_kick 40 250` |
| `config set default_speed` | speed (0-63) | `config set default_speed 28` |

### Sensor Commands

| Command | Parameters | Example |
|---------|-----------|---------|
| `sensor_loop set mode` | mode (stop/infinite/counted), [iterations] | `sensor_loop set mode infinite` |
| `sensor_loop set mode counted` | iterations | `sensor_loop set mode counted 50` |
| `sensor_loop set delay` | delay_ms (50-10000) | `sensor_loop set delay 500` |

### Calibration Commands

| Command | Parameters | Example |
|---------|-----------|---------|
| `calibrate gyro` | none | `calibrate gyro` |
| `calibrate mag` | none | `calibrate mag` |

### System Commands

| Command | Parameters | Example |
|---------|-----------|---------|
| `reboot` | none | `reboot` |

---

## Best Practices

1. **Always verify connection before running sequences:**
   ```python
   client = connect_broker()
   show_config(client)  # Verify robot responds
   ```

2. **Use calibration for straight movement:**
   ```python
   set_calibration(client, left=1.05, right=1.0)
   drive_forward(client)
   ```

3. **Handle keyboard interrupts gracefully:**
   ```python
   try:
       drive_forward(client, duration=10000)
   except KeyboardInterrupt:
       stop(client)
   ```

4. **Test with verbose output first:**
   ```python
   ENABLE_VERBOSE = True  # See what's happening
   ```

5. **Use appropriate speeds:**
   - **Precise movements:** SLOW_SPEED (15)
   - **Normal movements:** DRIVE_SPEED (32)
   - **Quick movements:** FAST_SPEED (45)

6. **Allow time after movements:**
   - Don't queue commands too quickly
   - Use SAFETY_BUFFER for extra margin
   - Wait for status messages

---

## Tips & Tricks

### Adjust Timing Without Code Changes

Edit the duration variables:
```python
# For slower movements
DRIVE_FORWARD_DURATION = 5000  # 5 seconds instead of 3.5
TURN_DURATION = 1500           # 1.5 seconds instead of 1
```

### Control Multiple Robots

Create separate connections:
```python
client1 = connect_broker()
SPARKNODE_ID = "sparknode01"

client2 = connect_broker()
SPARKNODE_ID = "sparknode02"

drive_forward(client1)  # Sparknode01 moves
drive_forward(client2)  # Sparknode02 moves
```

### Log to File

```python
import logging

logging.basicConfig(filename='sparknode.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Add to send_command():
logger.info(f"Sent: {command}")
```

### Create Custom Sequences

```python
def figure_eight(client, scale=1.0):
    """Execute figure-8 pattern"""
    duration = int(2000 * scale)
    
    for _ in range(2):  # Two loops
        turn_right(client, duration=duration)
        drive_forward(client, duration=duration)
        turn_right(client, duration=duration)
        
        turn_left(client, duration=duration)
        drive_forward(client, duration=duration)
        turn_left(client, duration=duration)

# Use it:
figure_eight(client, scale=1.5)
```

---

## Support & Debugging

For connection issues, enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check MQTT logs on Raspberry Pi:
```bash
sudo tail -f /var/log/mosquitto/mosquitto.log
```

Monitor robot directly:
```bash
mosquitto_sub -h 192.168.11.100 -t "arena/sparknode07/status"
```

---

**Last Updated:** November 2025  
**Version:** 1.0  
**Python:** 3.7+  
**Dependencies:** paho-mqtt
