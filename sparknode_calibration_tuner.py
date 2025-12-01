#!/usr/bin/env python3
"""
SparkNode Calibration Tuner
Interactive tool to calibrate and tune parameters for a specific SparkNode robot.
Allows real-time testing and adjustment of:
- Wheel speed calibration (left/right multipliers)
- Default speeds (drive, turn, rotation)
- Kick parameters (drive_kick, turn_kick)
- Sensor settings

Once satisfied, prints the final configuration with labels for copy-paste into the main script.
"""

import paho.mqtt.client as mqtt
import time
import sys
from typing import Optional, Dict, Tuple

# ============================================================================
# INITIAL CONFIGURATION - Change these before running
# ============================================================================

MQTT_BROKER = "192.168.11.100"
# MQTT_BROKER = "192.168.20.5"
MQTT_BROKER_PORT = 1883
SPARKNODE_ID = "sparknode07"  # Change this to calibrate different robots

# ============================================================================
# CALIBRATION STATE (Modified during tuning)
# ============================================================================

calibration_state = {
    "left_multiplier": 1.0,
    "right_multiplier": 1.19,
    "drive_speed": 25,
    "turn_speed": 14,
    "rotation_speed": 25,
    "drive_kick_speed": 25,
    "drive_kick_duration": 100,
    "turn_kick_speed": 40,
    "turn_kick_duration": 250,
    "sensor_delay": 500,
}

# ============================================================================
# MQTT SETUP
# ============================================================================

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_BROKER_PORT}\n")
    else:
        print(f"✗ Failed to connect, return code {rc}")
        sys.exit(1)

def on_message(client, userdata, msg):
    print(f"  Status: {msg.payload.decode()}")

def connect_broker() -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_BROKER_PORT, 60)
        client.loop_start()
        time.sleep(0.5)
        return client
    except Exception as e:
        print(f"✗ Connection error: {e}")
        sys.exit(1)

def send_command(client: mqtt.Client, command: str) -> None:
    try:
        topic = f"arena/{SPARKNODE_ID}/cmd"
        client.publish(topic, command, qos=1)
        print(f"  → Sent: {command}")
    except Exception as e:
        print(f"✗ Failed to send command: {e}")

def apply_calibration(client: mqtt.Client, state: Dict) -> None:
    """Apply current calibration state to robot"""
    send_command(client, f"config set calibration {state['left_multiplier']} {state['right_multiplier']}")
    time.sleep(0.3)
    send_command(client, f"config set drive_kick {state['drive_kick_speed']} {state['drive_kick_duration']}")
    time.sleep(0.3)
    send_command(client, f"config set turn_kick {state['turn_kick_speed']} {state['turn_kick_duration']}")
    time.sleep(0.3)

# ============================================================================
# UI FUNCTIONS
# ============================================================================

def print_header():
    print("\n" + "="*70)
    print("  SparkNode Calibration Tuner".center(70))
    print(f"  Target: {SPARKNODE_ID}".center(70))
    print("="*70 + "\n")

def print_current_state(state: Dict):
    print("\n┌─ Current Calibration State ─────────────────────────────────────┐")
    print(f"│ Left Motor Multiplier:      {state['left_multiplier']:<6.2f}                        │")
    print(f"│ Right Motor Multiplier:     {state['right_multiplier']:<6.2f}                        │")
    print(f"│ Drive Speed (0-63):         {state['drive_speed']:<3}                           │")
    print(f"│ Turn Speed (0-63):          {state['turn_speed']:<3}                           │")
    print(f"│ Rotation Speed (0-63):      {state['rotation_speed']:<3}                           │")
    print(f"│ Drive Kick Speed (16-63):   {state['drive_kick_speed']:<3}                           │")
    print(f"│ Drive Kick Duration (50-500ms): {state['drive_kick_duration']:<3}                   │")
    print(f"│ Turn Kick Speed (16-63):    {state['turn_kick_speed']:<3}                           │")
    print(f"│ Turn Kick Duration (50-500ms):  {state['turn_kick_duration']:<3}                    │")
    print(f"│ Sensor Delay (50-10000ms):  {state['sensor_delay']:<5}                        │")
    print("└────────────────────────────────────────────────────────────────┘\n")

def print_menu():
    print("┌─ Adjustment Menu ──────────────────────────────────────────────┐")
    print("│  Calibration Options:                                          │")
    print("│    1. Adjust Left Motor Multiplier (0.5-2.0)                  │")
    print("│    2. Adjust Right Motor Multiplier (0.5-2.0)                 │")
    print("│                                                                │")
    print("│  Speed Options:                                                │")
    print("│    3. Adjust Drive Speed (0-63)                               │")
    print("│    4. Adjust Turn Speed (0-63)                                │")
    print("│    5. Adjust Rotation Speed (0-63)                            │")
    print("│                                                                │")
    print("│  Kick Parameters:                                              │")
    print("│    6. Adjust Drive Kick Speed & Duration                      │")
    print("│    7. Adjust Turn Kick Speed & Duration                       │")
    print("│                                                                │")
    print("│  Testing:                                                      │")
    print("│    8. Test Forward Drive                                       │")
    print("│    9. Test Turn (Right)                                        │")
    print("│    10. Test Turn (Left)                                        │")
    print("│    11. Test Rotation (0° → 90° → 180° → 270°)                │")
    print("│    12. Emergency Stop                                          │")
    print("│                                                                │")
    print("│  Admin:                                                        │")
    print("│    13. Show Robot Config                                       │")
    print("│    14. Finish & Review                                         │")
    print("│    15. Exit (Discard Changes)                                  │")
    print("└────────────────────────────────────────────────────────────────┘\n")

def get_float_input(prompt: str, min_val: float, max_val: float, current: float) -> Optional[float]:
    """Get validated float input from user"""
    while True:
        try:
            print(f"{prompt} (current: {current}, range: {min_val}-{max_val}): ", end="")
            val = float(input().strip())
            if min_val <= val <= max_val:
                return val
            else:
                print(f"  ✗ Value must be between {min_val} and {max_val}")
        except ValueError:
            print(f"  ✗ Invalid input. Please enter a number.")

def get_int_input(prompt: str, min_val: int, max_val: int, current: int) -> Optional[int]:
    """Get validated integer input from user"""
    while True:
        try:
            print(f"{prompt} (current: {current}, range: {min_val}-{max_val}): ", end="")
            val = int(input().strip())
            if min_val <= val <= max_val:
                return val
            else:
                print(f"  ✗ Value must be between {min_val} and {max_val}")
        except ValueError:
            print(f"  ✗ Invalid input. Please enter a number.")

def adjust_left_multiplier(client: mqtt.Client, state: Dict):
    new_val = get_float_input("\nEnter new Left Motor Multiplier", 0.5, 2.0, state['left_multiplier'])
    if new_val is not None:
        state['left_multiplier'] = new_val
        print(f"✓ Set to {new_val}")
        apply_calibration(client, state)

def adjust_right_multiplier(client: mqtt.Client, state: Dict):
    new_val = get_float_input("\nEnter new Right Motor Multiplier", 0.5, 2.0, state['right_multiplier'])
    if new_val is not None:
        state['right_multiplier'] = new_val
        print(f"✓ Set to {new_val}")
        apply_calibration(client, state)

def adjust_drive_speed(client: mqtt.Client, state: Dict):
    new_val = get_int_input("\nEnter new Drive Speed", 0, 63, state['drive_speed'])
    if new_val is not None:
        state['drive_speed'] = new_val
        print(f"✓ Set to {new_val}")

def adjust_turn_speed(client: mqtt.Client, state: Dict):
    new_val = get_int_input("\nEnter new Turn Speed", 0, 63, state['turn_speed'])
    if new_val is not None:
        state['turn_speed'] = new_val
        print(f"✓ Set to {new_val}")

def adjust_rotation_speed(client: mqtt.Client, state: Dict):
    new_val = get_int_input("\nEnter new Rotation Speed", 0, 63, state['rotation_speed'])
    if new_val is not None:
        state['rotation_speed'] = new_val
        print(f"✓ Set to {new_val}")

def adjust_drive_kick(client: mqtt.Client, state: Dict):
    print("\nAdjusting Drive Kick Parameters...")
    speed = get_int_input("Enter Kick Speed", 16, 63, state['drive_kick_speed'])
    if speed is not None:
        duration = get_int_input("Enter Kick Duration (ms)", 50, 500, state['drive_kick_duration'])
        if duration is not None:
            state['drive_kick_speed'] = speed
            state['drive_kick_duration'] = duration
            print(f"✓ Drive Kick set to speed={speed}, duration={duration}ms")
            apply_calibration(client, state)

def adjust_turn_kick(client: mqtt.Client, state: Dict):
    print("\nAdjusting Turn Kick Parameters...")
    speed = get_int_input("Enter Kick Speed", 16, 63, state['turn_kick_speed'])
    if speed is not None:
        duration = get_int_input("Enter Kick Duration (ms)", 50, 500, state['turn_kick_duration'])
        if duration is not None:
            state['turn_kick_speed'] = speed
            state['turn_kick_duration'] = duration
            print(f"✓ Turn Kick set to speed={speed}, duration={duration}ms")
            apply_calibration(client, state)

def test_forward(client: mqtt.Client, state: Dict):
    print(f"\n→ Testing forward drive at speed {state['drive_speed']} for 3 seconds...")
    send_command(client, f"drive forward {state['drive_speed']} 3000")
    time.sleep(4)
    print("✓ Test complete")

def test_turn_right(client: mqtt.Client, state: Dict):
    print(f"\n→ Testing right turn at speed {state['turn_speed']} for 1 second...")
    send_command(client, f"turn right {state['turn_speed']} 1000")
    time.sleep(2)
    print("✓ Test complete")

def test_turn_left(client: mqtt.Client, state: Dict):
    print(f"\n→ Testing left turn at speed {state['turn_speed']} for 1 second...")
    send_command(client, f"turn left {state['turn_speed']} 1000")
    time.sleep(2)
    print("✓ Test complete")

def test_rotation(client: mqtt.Client, state: Dict):
    print(f"\n→ Starting rotation test at speed {state['rotation_speed']}...")
    print("  Rotating to: 0° → 90° → 180° → 270° → 0°")
    
    send_command(client, f"sensor_loop set mode infinite")
    time.sleep(1)
    
    for angle in [0, 90, 180, 270, 0]:
        print(f"  → Rotating to {angle}°...")
        send_command(client, f"rotate to {angle} {state['rotation_speed']} 20 100 10.0")
        time.sleep(16)
    
    send_command(client, f"sensor_loop set mode stop")
    time.sleep(1)
    print("✓ Rotation test complete")

def emergency_stop(client: mqtt.Client):
    print("\n⚠  EMERGENCY STOP")
    send_command(client, "stop")
    time.sleep(0.5)
    print("✓ Robot stopped")

def show_robot_config(client: mqtt.Client):
    print("\n→ Requesting robot configuration...")
    send_command(client, "config show")
    time.sleep(1)

def print_final_review(state: Dict) -> bool:
    """Print final configuration and ask for confirmation"""
    print("\n" + "="*70)
    print("  FINAL CALIBRATION REVIEW".center(70))
    print("="*70)
    
    print("\n✓ Copy these values to sparknode_mqtt_control.py:\n")
    
    print("# Calibration Settings")
    print(f"CALIBRATION_LEFT = {state['left_multiplier']:<6.2f} # Left wheel calibration factor (0.5-2.0)")
    print(f"CALIBRATION_RIGHT = {state['right_multiplier']:<6.2f} # Right wheel calibration factor (0.5-2.0)\n")
    
    print("# Movement Speed Parameters (0-63)")
    print(f"DEFAULT_SPEED = {state['drive_speed']:<3}  # 0-63, default ~51% PWM")
    print(f"TURN_SPEED = {state['turn_speed']:<3}     # Speed for turns")
    print(f"DRIVE_SPEED = {state['drive_speed']:<3}    # Speed for forward/reverse")
    print(f"ROTATION_SPEED = {state['rotation_speed']:<3}       # Default rotation speed (0-63)\n")
    
    print("# Kick Parameters")
    print(f"DRIVE_KICK_SPEED = {state['drive_kick_speed']:<3}     # Drive kick speed (16-63)")
    print(f"DRIVE_KICK_DURATION = {state['drive_kick_duration']:<3} # Drive kick duration (50-500ms)")
    print(f"TURN_KICK_SPEED = {state['turn_kick_speed']:<3}      # Turn kick speed (16-63)")
    print(f"TURN_KICK_DURATION = {state['turn_kick_duration']:<3}  # Turn kick duration (50-500ms)\n")
    
    print("# Sensor Parameters")
    print(f"SENSOR_DELAY_MS = {state['sensor_delay']:<5}     # Delay between sensor readings (50-10000ms)\n")
    
    print("="*70)
    print("\nAre you satisfied with these calibration values?")
    print("(y/n): ", end="")
    
    response = input().strip().lower()
    return response in ['y', 'yes']

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print_header()
    
    client = connect_broker()
    
    # Subscribe to status messages
    client.subscribe(f"arena/{SPARKNODE_ID}/status")
    
    print(f"Robot target: {SPARKNODE_ID}")
    print("Starting interactive calibration...\n")
    
    try:
        while True:
            print_current_state(calibration_state)
            print_menu()
            
            choice = input("Select option (1-15): ").strip()
            
            if choice == "1":
                adjust_left_multiplier(client, calibration_state)
            elif choice == "2":
                adjust_right_multiplier(client, calibration_state)
            elif choice == "3":
                adjust_drive_speed(client, calibration_state)
            elif choice == "4":
                adjust_turn_speed(client, calibration_state)
            elif choice == "5":
                adjust_rotation_speed(client, calibration_state)
            elif choice == "6":
                adjust_drive_kick(client, calibration_state)
            elif choice == "7":
                adjust_turn_kick(client, calibration_state)
            elif choice == "8":
                test_forward(client, calibration_state)
            elif choice == "9":
                test_turn_right(client, calibration_state)
            elif choice == "10":
                test_turn_left(client, calibration_state)
            elif choice == "11":
                test_rotation(client, calibration_state)
            elif choice == "12":
                emergency_stop(client)
            elif choice == "13":
                show_robot_config(client)
            elif choice == "14":
                if print_final_review(calibration_state):
                    print("\n✓ Calibration confirmed! Save these values to your script.")
                    print("\nFinal values for reference:")
                    print(f"  Left: {calibration_state['left_multiplier']}")
                    print(f"  Right: {calibration_state['right_multiplier']}")
                    print(f"  Drive Speed: {calibration_state['drive_speed']}")
                    print(f"  Turn Speed: {calibration_state['turn_speed']}")
                    break
                else:
                    print("\nReturning to menu...\n")
            elif choice == "15":
                print("\nExiting without saving...")
                break
            else:
                print("✗ Invalid option. Please select 1-15.")
                
    except KeyboardInterrupt:
        print("\n\n⚠  Calibration interrupted by user")
        emergency_stop(client)
    
    finally:
        print("\nDisconnecting...")
        client.loop_stop()
        client.disconnect()
        print("✓ Done")

if __name__ == "__main__":
    main()
