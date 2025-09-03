import asyncio
import struct
import logging
import threading
import time
import configparser
from pathlib import Path
from typing import Optional, Dict, Any
from queue import Queue, Empty

# Try to import BLE server functionality
BLE_AVAILABLE = False
try:
    from bless import BlessServer, GATTAttributePermissions, GATTCharacteristicProperties
    BLE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  BLE server functionality not available: {e}")
    print("   Running in ANT+ data display mode only.")
    print("   To enable BLE server on Windows, you may need to install additional dependencies.")

# ANT+ imports
ANT_AVAILABLE = False
try:
    from openant.easy.node import Node
    from openant.devices import ANTPLUS_NETWORK_KEY
    from openant.devices.heart_rate import HeartRate
    from openant.devices.power_meter import PowerMeter
    ANT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ANT+ functionality not available: {e}")
    print("   Install openant: pip install openant")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FTMSBridge:
    # FTMS Service and Characteristics UUIDs
    FTMS_SERVICE_UUID = "00001826-0000-1000-8000-00805f9b34fb"
    INDOOR_BIKE_DATA_UUID = "00002ad2-0000-1000-8000-00805f9b34fb"
    FITNESS_MACHINE_FEATURE_UUID = "00002acc-0000-1000-8000-00805f9b34fb"
    FITNESS_MACHINE_CONTROL_UUID = "00002ad9-0000-1000-8000-00805f9b34fb"
    FITNESS_MACHINE_STATUS_UUID = "00002ada-0000-1000-8000-00805f9b34fb"
    
    def __init__(self):
        # Load configuration first
        self.config = self.load_config()
        self.simulation_mode = self.config.getboolean('ADVANCED', 'simulation_mode', fallback=False)
        
        # Check available functionality
        self.ble_available = BLE_AVAILABLE
        self.ant_available = ANT_AVAILABLE
        
        # Allow running in simulation mode even if other features aren't available
        if not self.ble_available and not self.ant_available and not self.simulation_mode:
            raise RuntimeError("Neither BLE server nor ANT+ functionality is available, and simulation mode is disabled. Please install required dependencies or enable simulation mode in config.ini")
        
        self.hr_data = 150  # Start with a reasonable default instead of 0
        self.power_data = 250  # Start with a reasonable default instead of 0
        self.cadence_data = 90  # Start with a reasonable default instead of 0
        self.speed_data = 0.0  # Keep this as it's not used anymore
        
        # Resistance/Trainer control data
        self.target_resistance = 0.0  # Percentage (0-100)
        self.target_power = 0  # Watts
        self.target_slope = 0.0  # Grade percentage
        self.trainer_status = 0x00  # FTMS status flags
        
        self.server: Optional[object] = None  # Can be BlessServer if available
        self.node: Optional[object] = None    # Can be ANT+ Node if available
        self.ant_thread: Optional[threading.Thread] = None
        self.data_queue = Queue()
        self.running = False
        
        # Configuration-based settings
        self.device_name = self.config.get('BLE_SETTINGS', 'device_name', fallback='ANT+ FTMS Trainer')
        self.update_interval = self.config.getfloat('BLE_SETTINGS', 'update_interval', fallback=1.0)
        
        # Setup logging
        self.setup_logging()
        
        # Log availability status
        logger.info(f"BLE Server: {'✅ Available' if self.ble_available else '❌ Not Available'}")
        logger.info(f"ANT+ Support: {'✅ Available' if self.ant_available else '❌ Not Available'}")
        logger.info(f"Simulation Mode: {'✅ Enabled' if self.simulation_mode else '❌ Disabled'}")
    
    def load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini"""
        config = configparser.ConfigParser()
        config_file = Path('config.ini')
        
        if config_file.exists():
            try:
                config.read(config_file)
                logger.info("Configuration loaded from config.ini")
            except Exception as e:
                logger.warning(f"Failed to load config.ini: {e}, using defaults")
        else:
            logger.info("config.ini not found, using default settings")
        
        return config
    
    def setup_logging(self):
        """Configure logging based on config settings"""
        log_level = self.config.get('LOGGING', 'log_level', fallback='INFO')
        log_to_file = self.config.getboolean('LOGGING', 'log_to_file', fallback=False)
        
        # Set log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        # Add file handler if requested
        if log_to_file:
            file_handler = logging.FileHandler('bridge.log')
            file_handler.setLevel(numeric_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
    def create_indoor_bike_data(self) -> bytes:
        """Create FTMS Indoor Bike Data packet with heart rate included"""
        # According to FTMS spec, create a unified trainer packet
        # Many implementations expect speed to be present even if zero
        flags = 0x0000
        flags |= 0x0002  # Average Speed present (required by many apps)
        flags |= 0x0004  # Instantaneous Cadence present  
        flags |= 0x0040  # Instantaneous Power present
        flags |= 0x0200  # Heart Rate present (unified trainer)
        
        data = struct.pack('<H', flags)  # Flags (2 bytes)
        
        # Fields must appear in this exact order per FTMS spec:
        # 1. Instantaneous Speed (2 bytes) - when flag 0x0002 is set
        # 2. Average Speed (2 bytes) - when flag 0x0002 is set  
        # 3. Instantaneous Cadence (2 bytes) - when flag 0x0004 is set
        # 4. Instantaneous Power (2 bytes) - when flag 0x0040 is set
        # 5. Heart Rate (1 byte) - when flag 0x0200 is set
        
        # Instantaneous Speed (km/h * 100) - 2 bytes - set to 0
        data += struct.pack('<H', 0)
        
        # Average Speed (km/h * 100) - 2 bytes - set to 0  
        data += struct.pack('<H', 0)
        
        # Instantaneous Cadence (RPM * 2) - 2 bytes, unsigned
        cadence_scaled = max(0, min(65534, int(self.cadence_data * 2)))
        data += struct.pack('<H', cadence_scaled)
        
        # Instantaneous Power (watts) - 2 bytes (unsigned per FTMS spec)
        power_clamped = max(0, min(65535, int(self.power_data)))
        data += struct.pack('<H', power_clamped)
        
        # Heart Rate (bpm) - 1 byte
        hr_clamped = max(0, min(255, int(self.hr_data)))
        data += struct.pack('<B', hr_clamped)
        
        # Debug: Log the packet contents
        logger.info(f"FTMS packet: flags=0x{flags:04X}, speed=0, cadence={self.cadence_data}rpm (scaled={cadence_scaled}), power={self.power_data}W, hr={self.hr_data}bpm, packet_size={len(data)} bytes, hex={data.hex()}")
        
        return bytearray(data)

    def ant_worker_thread(self):
        """Worker thread for ANT+ operations"""
        if not ANT_AVAILABLE:
            logger.error("ANT+ libraries not available")
            self.data_queue.put(('error', 'ANT+ libraries not installed'))
            return
            
        try:
            logger.info("Starting ANT+ worker thread")
            self.node = Node()
            self.node.set_network_key(0, ANTPLUS_NETWORK_KEY)
            
            # Heart Rate Monitor
            self.hrm = HeartRate(self.node)
            self.hrm.on_device_data = self.on_hr_data
            
            # Power Meter (Smart Trainer)
            self.pm = PowerMeter(self.node)
            self.pm.on_device_data = self.on_power_data
            
            # Start the ANT+ node (this is blocking)
            self.node.start()
            
        except Exception as e:
            logger.error(f"ANT+ worker thread error: {e}")
            self.data_queue.put(('error', str(e)))
    
    def setup_ant_devices(self):
        """Initialize ANT+ connections in separate thread"""
        try:
            self.running = True
            
            if self.simulation_mode:
                logger.info("Simulation mode enabled - generating fake data")
                self.ant_thread = threading.Thread(target=self.simulation_worker_thread, daemon=True)
            elif self.ant_available:
                logger.info("Starting ANT+ devices")
                self.ant_thread = threading.Thread(target=self.ant_worker_thread, daemon=True)
            else:
                logger.warning("ANT+ not available and simulation mode disabled")
                return False
            
            self.ant_thread.start()
            logger.info("Data thread started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup data source: {e}")
            return False
    
    def simulation_worker_thread(self):
        """Worker thread for simulation mode"""
        try:
            logger.info("Starting simulation worker thread")
            
            # Get simulation values from config
            sim_hr = self.config.getint('ADVANCED', 'sim_heart_rate', fallback=150)
            sim_power = self.config.getint('ADVANCED', 'sim_power', fallback=250)
            sim_cadence = self.config.getint('ADVANCED', 'sim_cadence', fallback=90)
            
            logger.info(f"Simulation data: HR={sim_hr}, Power={sim_power}W, Cadence={sim_cadence}rpm")
            
            while self.running:
                # Simulate slight variations in data
                import random
                
                # Update with small random variations
                self.hr_data = max(60, min(200, sim_hr + random.randint(-5, 5)))
                self.power_data = max(0, sim_power + random.randint(-20, 20))
                self.cadence_data = max(0, sim_cadence + random.randint(-5, 5))
                # Removed speed simulation
                
                # Debug: Log the actual values being set
                logger.info(f"Simulation: Updated values - HR: {self.hr_data}, Power: {self.power_data}, Cadence: {self.cadence_data}")
                
                # Ensure values are not zero
                if self.hr_data == 0:
                    self.hr_data = 150
                if self.power_data == 0:
                    self.power_data = 250
                if self.cadence_data == 0:
                    self.cadence_data = 90
                
                # Put data in queue for logging
                self.data_queue.put(('hr', self.hr_data))
                self.data_queue.put(('power', {
                    'power': self.power_data,
                    'cadence': self.cadence_data
                }))
                
                time.sleep(1)  # Update every second
                
        except Exception as e:
            logger.error(f"Simulation worker thread error: {e}")
            self.data_queue.put(('error', str(e)))
    
    def on_hr_data(self, data: Dict[str, Any]):
        """Handle incoming heart rate data"""
        try:
            if 'heart_rate' in data:
                self.hr_data = int(data['heart_rate'])
                logger.debug(f"HR: {self.hr_data} bpm")
                self.data_queue.put(('hr', self.hr_data))
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid HR data: {data}, error: {e}")
    
    def on_power_data(self, data: Dict[str, Any]):
        """Handle incoming power meter data"""
        try:
            updated = False
            if 'power' in data:
                self.power_data = int(data['power'])
                updated = True
            if 'cadence' in data:
                self.cadence_data = int(data['cadence'])
                updated = True
            # Removed speed handling - not using speed data
            
            if updated:
                logger.debug(f"Power: {self.power_data}W, Cadence: {self.cadence_data}rpm")
                self.data_queue.put(('power', {
                    'power': self.power_data,
                    'cadence': self.cadence_data
                }))
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid power data: {data}, error: {e}")
    
    def handle_ftms_control_point(self, data: bytearray) -> bytearray:
        """Handle FTMS Control Point commands from Zwift"""
        if len(data) == 0:
            return bytearray([0x80, 0x01, 0x02])  # Error response
        
        opcode = data[0]
        logger.info(f"FTMS Control Point command: 0x{opcode:02X}, data: {data.hex()}")
        
        # FTMS Control Point opcodes
        if opcode == 0x00:  # Request Control
            logger.info("Zwift requesting control")
            self.trainer_status |= 0x01  # Set control granted
            return bytearray([0x80, 0x00, 0x01])  # Success response
            
        elif opcode == 0x01:  # Reset
            logger.info("Zwift requesting reset")
            self.target_resistance = 0.0
            self.target_power = 0
            self.target_slope = 0.0
            return bytearray([0x80, 0x01, 0x01])  # Success response
            
        elif opcode == 0x04:  # Set Target Resistance Level
            if len(data) >= 3:
                resistance = int.from_bytes(data[1:3], byteorder='little') / 10.0  # 0.1% resolution
                self.target_resistance = min(100.0, max(0.0, resistance))
                logger.info(f"Zwift set target resistance: {self.target_resistance:.1f}%")
                return bytearray([0x80, 0x04, 0x01])  # Success response
            
        elif opcode == 0x05:  # Set Target Power
            if len(data) >= 3:
                power = int.from_bytes(data[1:3], byteorder='little')
                self.target_power = max(0, power)
                logger.info(f"Zwift set target power: {self.target_power}W")
                return bytearray([0x80, 0x05, 0x01])  # Success response
                
        elif opcode == 0x07:  # Start or Resume
            logger.info("Zwift start/resume command")
            self.trainer_status |= 0x04  # Set started
            return bytearray([0x80, 0x07, 0x01])  # Success response
            
        elif opcode == 0x08:  # Stop or Pause
            logger.info("Zwift stop/pause command")
            self.trainer_status &= ~0x04  # Clear started
            return bytearray([0x80, 0x08, 0x01])  # Success response
            
        elif opcode == 0x11:  # Set Indoor Bike Simulation Parameters
            if len(data) >= 7:
                wind_speed = int.from_bytes(data[1:3], byteorder='little', signed=True) / 1000.0  # m/s
                grade = int.from_bytes(data[3:5], byteorder='little', signed=True) / 100.0  # %
                rolling_resistance = data[5] / 10000.0  # Coefficient
                wind_resistance = data[6] / 100.0  # kg/m
                
                self.target_slope = grade
                logger.info(f"Zwift simulation: Grade={grade:.1f}%, Wind={wind_speed:.1f}m/s, RR={rolling_resistance:.4f}, WR={wind_resistance:.2f}")
                return bytearray([0x80, 0x11, 0x01])  # Success response
        
        # Unknown or unsupported command
        logger.warning(f"Unsupported FTMS control command: 0x{opcode:02X}")
        return bytearray([0x80, opcode, 0x02])  # Not supported response
    
    def create_ftms_status_data(self) -> bytearray:
        """Create FTMS Status packet"""
        # Simple status packet - just return current trainer status
        return bytearray([self.trainer_status])

    def get_fitness_machine_features_bytes(self) -> bytearray:
        """Return supported fitness machine features as bytes"""
        # Feature flags indicating supported features
        features = 0x00000000
        features |= 0x00000001  # Average Speed Supported
        features |= 0x00000002  # Cadence Supported
        features |= 0x00000008  # Power Measurement Supported
        features |= 0x00008000  # Indoor Bike Simulation Parameters Supported
        features |= 0x00004000  # Resistance Level Supported
        features |= 0x00002000  # Target Setting Supported
        
        target_settings = 0x00000000
        target_settings |= 0x00000001  # Target Resistance Level Supported
        target_settings |= 0x00000002  # Target Power Supported
        target_settings |= 0x00000008  # Indoor Bike Simulation Parameters Supported
        
        return bytearray(struct.pack('<LL', features, target_settings))

    async def fitness_machine_features_read(self, **kwargs) -> bytearray:
        """Return supported fitness machine features (async version for compatibility)"""
        return self.get_fitness_machine_features_bytes()
    
    async def setup_ble_server(self):
        """Setup BLE GATT server with FTMS services"""
        if not self.ble_available:
            logger.warning("BLE server functionality not available - skipping BLE setup")
            return False
            
        try:
            logger.info("Creating BLE server...")
            # Create the BLE server
            self.server = BlessServer(name=self.device_name)
            
            logger.info("Adding FTMS service...")
            # Add FTMS Service
            ftms_service_uuid = self.FTMS_SERVICE_UUID
            await self.server.add_new_service(ftms_service_uuid)
            
            logger.info("Adding fitness machine feature characteristic...")
            # Add Fitness Machine Feature Characteristic (read-only)
            await self.server.add_new_characteristic(
                ftms_service_uuid,
                self.FITNESS_MACHINE_FEATURE_UUID,
                GATTCharacteristicProperties.read,
                self.get_fitness_machine_features_bytes(),
                GATTAttributePermissions.readable
            )
            
            logger.info("Adding indoor bike data characteristic...")
            # Add Indoor Bike Data Characteristic (notify and indicate)
            # Create initial FTMS packet with current values
            initial_ftms_data = self.create_indoor_bike_data()
            await self.server.add_new_characteristic(
                ftms_service_uuid,
                self.INDOOR_BIKE_DATA_UUID,
                GATTCharacteristicProperties.notify | GATTCharacteristicProperties.indicate,
                initial_ftms_data,  # Use proper FTMS packet as initial value
                GATTAttributePermissions.readable
            )
            
            logger.info("Adding FTMS control point characteristic...")
            # Add FTMS Control Point Characteristic (write with response, indicate)
            await self.server.add_new_characteristic(
                ftms_service_uuid,
                self.FITNESS_MACHINE_CONTROL_UUID,
                GATTCharacteristicProperties.write | GATTCharacteristicProperties.indicate,
                bytearray([0x00]),  # Initial value
                GATTAttributePermissions.writeable
            )
            
            logger.info("Adding FTMS status characteristic...")
            # Add FTMS Status Characteristic (notify)
            initial_status = self.create_ftms_status_data()
            await self.server.add_new_characteristic(
                ftms_service_uuid,
                self.FITNESS_MACHINE_STATUS_UUID,
                GATTCharacteristicProperties.notify,
                initial_status,
                GATTAttributePermissions.readable
            )

            logger.info("Setting up read callback...")
            # Set up read callback function for any read requests
            def read_callback(characteristic, **kwargs):
                """Handle read requests for characteristics"""
                # Extract the UUID from the characteristic object
                char_uuid = str(characteristic.uuid) if hasattr(characteristic, 'uuid') else str(characteristic)
                logger.info(f"Client reading characteristic: {char_uuid}")
                
                if char_uuid.upper() == self.FITNESS_MACHINE_FEATURE_UUID.upper():
                    features_data = self.get_fitness_machine_features_bytes()
                    logger.info(f"Returning fitness machine features: {features_data.hex()}")
                    return features_data
                else:
                    logger.info(f"Unknown characteristic read request: {char_uuid}")
                    # Return empty data for other characteristics
                    return bytearray()
            
            # Set up write callback function for control commands
            def write_callback(characteristic, data, **kwargs):
                """Handle write requests for characteristics"""
                char_uuid = str(characteristic.uuid) if hasattr(characteristic, 'uuid') else str(characteristic)
                logger.info(f"Client writing to characteristic: {char_uuid}, data: {data.hex() if data else 'None'}")
                
                if char_uuid.upper() == self.FITNESS_MACHINE_CONTROL_UUID.upper():
                    # Handle FTMS control point commands
                    if data:
                        response = self.handle_ftms_control_point(bytearray(data))
                        logger.info(f"FTMS control response: {response.hex()}")
                        
                        # Send response via indication
                        try:
                            control_char = self.server.get_characteristic(self.FITNESS_MACHINE_CONTROL_UUID)
                            if control_char:
                                control_char.value = response
                                self.server.update_value(self.FTMS_SERVICE_UUID, self.FITNESS_MACHINE_CONTROL_UUID)
                        except Exception as e:
                            logger.error(f"Failed to send control response: {e}")
                else:
                    logger.warning(f"Write to unknown characteristic: {char_uuid}")
            
            # Set the callbacks
            self.server.read_request_func = read_callback
            self.server.write_request_func = write_callback
            logger.info(f"Read and write callbacks set")

            logger.info("Starting BLE server...")
            # Start the BLE server with timeout
            try:
                await asyncio.wait_for(self.server.start(), timeout=15.0)
                logger.info(f"BLE FTMS server '{self.device_name}' started successfully")
                
                # Verify the callback is still set after starting
                logger.info(f"Read callback after start: {callable(self.server.read_request_func)}")
                return True
            except asyncio.TimeoutError:
                logger.error("BLE server start timeout (15s) - this usually indicates a permission issue")
                logger.error("On Windows, you may need to:")
                logger.error("  1. Run as administrator (right-click -> 'Run as administrator')")
                logger.error("  2. Or use run_admin.bat")
                logger.error("  3. Check Windows Bluetooth settings")
                return False
            
        except Exception as e:
            logger.error(f"Failed to setup BLE server: {e}")
            logger.debug(f"BLE setup error details: {type(e).__name__}: {str(e)}")
            if "access" in str(e).lower() or "permission" in str(e).lower():
                logger.error("BLE permission error - try running as administrator")
            return False
    
    async def notify_clients(self):
        """Notify BLE clients with latest sensor data"""
        if not self.ble_available or not self.server:
            return
            
        try:
            # Check if server is running (advertising) - try to update values regardless
            # Update Indoor Bike Data characteristic
            indoor_data = self.create_indoor_bike_data()
            logger.debug(f"Sending FTMS data - Power: {self.power_data}W, Cadence: {self.cadence_data}rpm")
            try:
                # Get the characteristic object and set its value directly
                ftms_char = self.server.get_characteristic(self.INDOOR_BIKE_DATA_UUID)
                if ftms_char:
                    ftms_char.value = bytearray(indoor_data)
                    # Then notify subscribed clients (not async)
                    success = self.server.update_value(
                        self.FTMS_SERVICE_UUID,
                        self.INDOOR_BIKE_DATA_UUID
                    )
                    if success:
                        logger.debug("FTMS characteristic updated successfully")
                    else:
                        logger.warning("FTMS characteristic update returned False")
                else:
                    logger.error("FTMS characteristic not found")
            except Exception as e:
                logger.error(f"Failed to update FTMS characteristic: {e}")
            
            # Update FTMS Status characteristic (less frequently)
            try:
                status_data = self.create_ftms_status_data()
                status_char = self.server.get_characteristic(self.FITNESS_MACHINE_STATUS_UUID)
                if status_char:
                    status_char.value = status_data
                    success = self.server.update_value(
                        self.FTMS_SERVICE_UUID,
                        self.FITNESS_MACHINE_STATUS_UUID
                    )
                    if success:
                        logger.debug("FTMS status updated successfully")
            except Exception as e:
                logger.debug(f"Failed to update FTMS status: {e}")
                    
        except Exception as e:
            logger.debug(f"Error notifying clients (normal if no clients connected): {e}")
    
    def display_data(self):
        """Display current sensor data (when BLE is not available)"""
        resistance_info = f" | Target: {self.target_resistance:.1f}% / {self.target_power}W / {self.target_slope:.1f}%" if self.target_resistance > 0 or self.target_power > 0 or abs(self.target_slope) > 0.1 else ""
        print(f"\r🚴 HR: {self.hr_data:3d} bpm | Power: {self.power_data:3d}W | Cadence: {self.cadence_data:3d} rpm{resistance_info}", end="", flush=True)
    
    async def run(self):
        """Main execution loop"""
        try:
            # Initialize data source (ANT+ or simulation)
            if not self.setup_ant_devices():
                logger.error("Failed to setup data source")
                return
            
            # Give data source time to initialize
            await asyncio.sleep(2)
            
            # Setup BLE server (if available)
            ble_started = False
            if self.ble_available:
                ble_started = await self.setup_ble_server()
                if ble_started:
                    logger.info("FTMS Bridge running with BLE server... Press Ctrl+C to stop")
                else:
                    logger.warning("BLE server failed to start, running in display-only mode")
            else:
                logger.info("Running in display-only mode (BLE not available)... Press Ctrl+C to stop")
            
            # Main loop - process and display/broadcast data
            data_counter = 0
            while self.running:
                try:
                    # Process any data from the queue
                    data_updated = False
                    while True:
                        try:
                            data_type, data_value = self.data_queue.get_nowait()
                            if data_type == 'error':
                                logger.error(f"Data source error: {data_value}")
                            elif data_type == 'hr':
                                # Update HR data from queue (for ANT+ mode)
                                if not self.simulation_mode:  # Only update if not in simulation
                                    self.hr_data = data_value
                                data_updated = True
                            elif data_type == 'power':
                                # Update power data from queue (for ANT+ mode)
                                if not self.simulation_mode:  # Only update if not in simulation
                                    self.power_data = data_value.get('power', self.power_data)
                                    self.cadence_data = data_value.get('cadence', self.cadence_data)
                                data_updated = True
                            else:
                                data_updated = True
                        except Empty:
                            break
                    
                    # Update display/notifications
                    if self.ble_available and ble_started:
                        # Debug: Show current values before sending
                        if data_counter % 5 == 0:  # Every 5 seconds
                            logger.debug(f"Current values before BLE update: HR={self.hr_data}, Power={self.power_data}, Cadence={self.cadence_data}")
                        
                        # Notify BLE clients with latest data
                        await self.notify_clients()
                    else:
                        # Display data to console
                        self.display_data()
                    
                    # Periodically log data when BLE is working
                    if ble_started and data_counter % 10 == 0:  # Log every 10 seconds
                        logger.info(f"Broadcasting - HR: {self.hr_data} bpm, Power: {self.power_data}W, Cadence: {self.cadence_data} rpm")
                    
                    data_counter += 1
                    
                    # Wait before next update
                    await asyncio.sleep(self.update_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1.0)
                    
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Starting cleanup...")
        self.running = False
        
        # Stop BLE server
        if self.server:
            try:
                await self.server.stop()
                logger.info("BLE server stopped")
            except Exception as e:
                logger.error(f"Error stopping BLE server: {e}")
        
        # Stop ANT+ node
        if self.node:
            try:
                self.node.stop()
                logger.info("ANT+ node stopped")
            except Exception as e:
                logger.error(f"Error stopping ANT+ node: {e}")
        
        # Wait for ANT+ thread to finish
        if self.ant_thread and self.ant_thread.is_alive():
            try:
                self.ant_thread.join(timeout=5)
                if self.ant_thread.is_alive():
                    logger.warning("ANT+ thread did not finish within timeout")
                else:
                    logger.info("ANT+ thread finished")
            except Exception as e:
                logger.error(f"Error joining ANT+ thread: {e}")
        
        logger.info("Cleanup completed")

if __name__ == "__main__":
    bridge = FTMSBridge()
    asyncio.run(bridge.run())