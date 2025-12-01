#!/usr/bin/env python3
"""
SparkNode MQTT Control Script
Sends movement commands to SparkNode robots via MQTT
Includes configurable variables for easy customization
"""

import paho.mqtt.client as mqtt
import time
import sys
from typing import Optional

# ============================================================================
# CONFIGURATION VARIABLES - Customize these values
# ============================================================================

# MQTT Broker Settings
MQTT_BROKER = "192.168.11.100"
# MQTT_BROKER = "192.168.20.5"
MQTT_BROKER_PORT = 1883
MQTT_KEEPALIVE = 60

# Target SparkNode
SPARKNODE_ID = "sparknode07"
MQTT_TOPIC_CMD = f"arena/{SPARKNODE_ID}/cmd"
MQTT_TOPIC_STATUS = f"arena/{SPARKNODE_ID}/status"

# Movement Speed Parameters (0-63)
DEFAULT_SPEED = 32  # 0-63, default ~51% PWM
TURN_SPEED = 14     # Speed for turns
DRIVE_SPEED = 25    # Speed for forward/reverse
SLOW_SPEED = 15     # Slow speed for precision movements
FAST_SPEED = 45     # Fast speed for quick movements

# Duration Multipliers (all in milliseconds)
TURN_DURATION = 1000          # 1 second turn
DRIVE_FORWARD_DURATION = 3000 # 3 seconds forward
DRIVE_REVERSE_DURATION = 2000 # 2 seconds reverse
SHORT_WAIT = 2000             # 2 second wait between commands

# Calibration Settings
CALIBRATION_LEFT = 1.0   # Left wheel calibration factor (0.5-2.0)
CALIBRATION_RIGHT = 1.19  # Right wheel calibration factor (0.5-2.0)

# Kick Parameters
DRIVE_KICK_SPEED = 25     # Drive kick speed (16-63)
DRIVE_KICK_DURATION = 100 # Drive kick duration (50-500ms)
TURN_KICK_SPEED = 40      # Turn kick speed (16-63)
TURN_KICK_DURATION = 250  # Turn kick duration (50-500ms)

# Rotation Parameters
ROTATION_SPEED = 25       # Default rotation speed (0-63)
ROTATION_STEP_MS = 20     # Duration of each rotation step (ms)
ROTATION_DELAY_MS = 100   # Delay between heading checks (ms)
ROTATION_TOLERANCE = 10.0 # Acceptable heading error (degrees)

# Sensor Parameters
SENSOR_DELAY_MS = 500     # Delay between sensor readings (50-10000ms)

# Script Behavior
ENABLE_VERBOSE = True     # Print detailed output
BREAK_IN_CYCLES = 5       # Number of break-in cycles to run
SAFETY_BUFFER = 1.0       # Safety buffer multiplier for sleep times (1.0 = command_duration + 1 sec)

# ============================================================================
# MQTT CALLBACK FUNCTIONS
# ============================================================================

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        if ENABLE_VERBOSE:
            print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_BROKER_PORT}")
        client.subscribe(MQTT_TOPIC_STATUS)
    else:
        print(f"✗ Failed to connect, return code {rc}")
        sys.exit(1)

def on_disconnect(client, userdata, rc):
    """Callback for when the client disconnects from the broker"""
    if rc != 0:
        print(f"✗ Unexpected disconnection. Return code: {rc}")
    else:
        if ENABLE_VERBOSE:
            print("✓ Disconnected from MQTT broker")

def on_message(client, userdata, msg):
    """Callback for when a message is received from the broker"""
    if ENABLE_VERBOSE:
        print(f"  Status: {msg.payload.decode()}")

def on_publish(client, userdata, mid):
    """Callback for when a message is successfully published"""
    pass

# ============================================================================
# MQTT HELPER FUNCTIONS
# ============================================================================

def connect_broker() -> mqtt.Client:
    """Connect to MQTT broker and return client"""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.on_publish = on_publish
    
    try:
        client.connect(MQTT_BROKER, MQTT_BROKER_PORT, MQTT_KEEPALIVE)
        client.loop_start()  # Start background thread
        time.sleep(0.5)  # Brief pause for connection to establish
        return client
    except Exception as e:
        print(f"✗ Connection error: {e}")
        sys.exit(1)

def send_command(client: mqtt.Client, command: str) -> None:
    """Send MQTT command to SparkNode"""
    try:
        client.publish(MQTT_TOPIC_CMD, command, qos=1)
        if ENABLE_VERBOSE:
            print(f"  → Sent: {command}")
    except Exception as e:
        print(f"✗ Failed to send command: {e}")

def calculate_sleep_time(duration_ms: int, multiplier: float = SAFETY_BUFFER) -> float:
    """Calculate sleep time based on command duration and safety buffer"""
    return (duration_ms / 1000.0) + multiplier

# ============================================================================
# MOVEMENT FUNCTIONS
# ============================================================================

def drive_forward(client: mqtt.Client, speed: Optional[int] = None, duration: Optional[int] = None) -> None:
    """Drive forward at specified speed for duration"""
    spd = speed or DRIVE_SPEED
    dur = duration or DRIVE_FORWARD_DURATION
    command = f"drive forward {spd} {dur}"
    send_command(client, command)
    sleep_time = calculate_sleep_time(dur)
    if ENABLE_VERBOSE:
        print(f"  ⏱  Waiting {sleep_time:.1f}s...")
    time.sleep(sleep_time)

def drive_reverse(client: mqtt.Client, speed: Optional[int] = None, duration: Optional[int] = None) -> None:
    """Drive in reverse at specified speed for duration"""
    spd = speed or DRIVE_SPEED
    dur = duration or DRIVE_REVERSE_DURATION
    command = f"drive reverse {spd} {dur}"
    send_command(client, command)
    sleep_time = calculate_sleep_time(dur)
    if ENABLE_VERBOSE:
        print(f"  ⏱  Waiting {sleep_time:.1f}s...")
    time.sleep(sleep_time)

def turn_left(client: mqtt.Client, speed: Optional[int] = None, duration: Optional[int] = None) -> None:
    """Turn left at specified speed for duration"""
    spd = speed or TURN_SPEED
    dur = duration or TURN_DURATION
    command = f"turn left {spd} {dur}"
    send_command(client, command)
    sleep_time = calculate_sleep_time(dur)
    if ENABLE_VERBOSE:
        print(f"  ⏱  Waiting {sleep_time:.1f}s...")
    time.sleep(sleep_time)

def turn_right(client: mqtt.Client, speed: Optional[int] = None, duration: Optional[int] = None) -> None:
    """Turn right at specified speed for duration"""
    spd = speed or TURN_SPEED
    dur = duration or TURN_DURATION
    command = f"turn right {spd} {dur}"
    send_command(client, command)
    sleep_time = calculate_sleep_time(dur)
    if ENABLE_VERBOSE:
        print(f"  ⏱  Waiting {sleep_time:.1f}s...")
    time.sleep(sleep_time)

def rotate_to(client: mqtt.Client, angle: int, speed: Optional[int] = None,
              step_ms: Optional[int] = None, delay_ms: Optional[int] = None,
              tolerance: Optional[float] = None) -> None:
    """Rotate to specific heading angle (0-360 degrees)"""
    spd = speed or ROTATION_SPEED
    step = step_ms or ROTATION_STEP_MS
    delay = delay_ms or ROTATION_DELAY_MS
    tol = tolerance or ROTATION_TOLERANCE
    
    command = f"rotate to {angle} {spd} {step} {delay} {tol}"
    send_command(client, command)
    if ENABLE_VERBOSE:
        print(f"  ⏱  Waiting ~15s for rotation...")
    time.sleep(15)  # Allow time for rotation to complete

def stop(client: mqtt.Client) -> None:
    """Stop all motors immediately"""
    command = "stop"
    send_command(client, command)
    if ENABLE_VERBOSE:
        print("  🛑 STOP command sent")
    time.sleep(0.5)

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def set_calibration(client: mqtt.Client, left: Optional[float] = None, right: Optional[float] = None) -> None:
    """Set wheel speed calibration factors"""
    l = left or CALIBRATION_LEFT
    r = right or CALIBRATION_RIGHT
    command = f"config set calibration {l} {r}"
    send_command(client, command)
    time.sleep(0.5)

def set_drive_kick(client: mqtt.Client, speed: Optional[int] = None, duration: Optional[int] = None) -> None:
    """Set drive kick parameters"""
    spd = speed or DRIVE_KICK_SPEED
    dur = duration or DRIVE_KICK_DURATION
    command = f"config set drive_kick {spd} {dur}"
    send_command(client, command)
    time.sleep(0.5)

def set_turn_kick(client: mqtt.Client, speed: Optional[int] = None, duration: Optional[int] = None) -> None:
    """Set turn kick parameters"""
    spd = speed or TURN_KICK_SPEED
    dur = duration or TURN_KICK_DURATION
    command = f"config set turn_kick {spd} {dur}"
    send_command(client, command)
    time.sleep(0.5)

def set_default_speed(client: mqtt.Client, speed: int) -> None:
    """Set default movement speed"""
    command = f"config set default_speed {speed}"
    send_command(client, command)
    time.sleep(0.5)

def show_config(client: mqtt.Client) -> None:
    """Display current configuration"""
    command = "config show"
    send_command(client, command)
    time.sleep(1)

# ============================================================================
# SENSOR FUNCTIONS
# ============================================================================

def start_sensor_loop(client: mqtt.Client, mode: str = "infinite", iterations: Optional[int] = None) -> None:
    """Start sensor data collection loop"""
    if mode == "counted" and iterations:
        command = f"sensor_loop set mode counted {iterations}"
    else:
        command = f"sensor_loop set mode {mode}"
    send_command(client, command)
    time.sleep(0.5)

def stop_sensor_loop(client: mqtt.Client) -> None:
    """Stop sensor data collection"""
    command = "sensor_loop set mode stop"
    send_command(client, command)
    time.sleep(0.5)

def set_sensor_delay(client: mqtt.Client, delay_ms: Optional[int] = None) -> None:
    """Set delay between sensor readings"""
    delay = delay_ms or SENSOR_DELAY_MS
    command = f"sensor_loop set delay {delay}"
    send_command(client, command)
    time.sleep(0.5)

# ============================================================================
# CALIBRATION FUNCTIONS
# ============================================================================

def calibrate_gyro(client: mqtt.Client) -> None:
    """Calibrate gyroscope"""
    command = "calibrate gyro"
    send_command(client, command)
    if ENABLE_VERBOSE:
        print("  ⏱  Gyro calibration in progress...")
    time.sleep(5)

def calibrate_mag(client: mqtt.Client) -> None:
    """Calibrate magnetometer"""
    command = "calibrate mag"
    send_command(client, command)
    if ENABLE_VERBOSE:
        print("  ⏱  Magnetometer calibration in progress...")
    time.sleep(10)

# ============================================================================
# SEQUENCE FUNCTIONS
# ============================================================================

def square_pattern_right(client: mqtt.Client, cycles: Optional[int] = None) -> None:
    """Execute square pattern turning right (break-in cycle)"""
    num_cycles = cycles or BREAK_IN_CYCLES
    
    print(f"\n{'='*60}")
    print(f"Starting Square Pattern (Right) - {num_cycles} cycles")
    print(f"{'='*60}\n")
    
    try:
        # Initial calibration
        print("[1/4] Setting initial calibration...")
        set_calibration(client, 1.0, 1.05)
        
        for cycle in range(1, num_cycles + 1):
            print(f"\n{'─'*60}")
            print(f"Cycle {cycle}/{num_cycles}")
            print(f"{'─'*60}")
            
            # Turn right
            print("\n[1/2] Turning right...")
            set_calibration(client, 1.0, 1.16)
            turn_right(client, speed=30, duration=TURN_DURATION)
            
            # Adjust castor wheels
            print("\n[2/2] Adjusting castor wheels & driving forward...")
            set_calibration(client, 1.0, 1.495)
            drive_forward(client, speed=10, duration=DRIVE_FORWARD_DURATION)
            
            print(f"\n✓ Cycle {cycle} complete")
    
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user")
        stop(client)
    finally:
        print(f"\n{'='*60}")
        print("Sequence complete")
        print(f"{'='*60}\n")

def square_pattern_left(client: mqtt.Client, cycles: Optional[int] = None) -> None:
    """Execute square pattern turning left (break-in cycle)"""
    num_cycles = cycles or BREAK_IN_CYCLES
    
    print(f"\n{'='*60}")
    print(f"Starting Square Pattern (Left) - {num_cycles} cycles")
    print(f"{'='*60}\n")
    
    try:
        # Initial calibration
        print("[1/4] Setting initial calibration...")
        set_calibration(client, 1.05, 1.0)
        
        for cycle in range(1, num_cycles + 1):
            print(f"\n{'─'*60}")
            print(f"Cycle {cycle}/{num_cycles}")
            print(f"{'─'*60}")
            
            # Turn left
            print("\n[1/2] Turning left...")
            set_calibration(client, 1.05, 1.0)
            turn_left(client, speed=30, duration=TURN_DURATION)
            
            # Adjust castor wheels
            print("\n[2/2] Adjusting castor wheels & driving forward...")
            set_calibration(client, 1.2, 1.0)
            drive_forward(client, speed=10, duration=DRIVE_FORWARD_DURATION)
            
            print(f"\n✓ Cycle {cycle} complete")
    
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user")
        stop(client)
    finally:
        print(f"\n{'='*60}")
        print("Sequence complete")
        print(f"{'='*60}\n")

def rotation_sequence(client: mqtt.Client) -> None:
    """Execute rotation through cardinal directions"""
    print(f"\n{'='*60}")
    print("Starting Cardinal Rotation Sequence")
    print(f"{'='*60}\n")
    
    try:
        # Start continuous sensor loop for heading feedback
        print("Starting sensor loop for heading feedback...")
        start_sensor_loop(client, mode="infinite")
        time.sleep(2)
        
        directions = [
            (0, "North"),
            (90, "East"),
            (180, "South"),
            (270, "West"),
            (0, "North (final)"),
        ]
        
        for angle, direction in directions:
            print(f"\n{'─'*60}")
            print(f"Rotating to {direction} ({angle}°)")
            print(f"{'─'*60}")
            rotate_to(client, angle, speed=ROTATION_SPEED)
        
        # Stop sensor loop
        print("\nStopping sensor loop...")
        stop_sensor_loop(client)
        
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user")
        stop(client)
        stop_sensor_loop(client)
    finally:
        print(f"\n{'='*60}")
        print("Rotation sequence complete")
        print(f"{'='*60}\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║        SparkNode MQTT Control Script (Python)            ║
╚══════════════════════════════════════════════════════════╝

Configuration:
  Broker: {MQTT_BROKER}:{MQTT_BROKER_PORT}
  Target: {SPARKNODE_ID}
  Default Speed: {DEFAULT_SPEED}
  Calibration: L={CALIBRATION_LEFT}, R={CALIBRATION_RIGHT}
  
Options:
  1. Square Pattern (Right)
  2. Square Pattern (Left)
  3. Cardinal Rotation Sequence
  4. Custom Commands (Interactive)
  5. Exit
""")
    
    client = connect_broker()
    
    try:
        while True:
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                square_pattern_right(client)
            elif choice == "2":
                square_pattern_left(client)
            elif choice == "3":
                rotation_sequence(client)
            elif choice == "4":
                print("\nEnter custom command (or 'exit' to return):")
                while True:
                    cmd = input(">> ").strip()
                    if cmd.lower() == "exit":
                        break
                    if cmd:
                        send_command(client, cmd)
                        time.sleep(1)
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid option. Please select 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user")
        stop(client)
    
    finally:
        print("Disconnecting...")
        client.loop_stop()
        client.disconnect()
        print("✓ Done")

if __name__ == "__main__":
    main()
