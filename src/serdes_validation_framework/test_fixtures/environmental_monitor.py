"""
Environmental Monitor

This module provides environmental monitoring capabilities for
controlled test environments in SerDes validation.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class EnvironmentalReading:
    """Single environmental reading"""
    timestamp: float
    temperature: float  # Celsius
    humidity: float     # Percentage
    pressure: float     # kPa
    vibration: float    # g-force
    noise_level: float  # dB
    air_flow: float     # m/s
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'vibration': self.vibration,
            'noise_level': self.noise_level,
            'air_flow': self.air_flow
        }


@dataclass
class EnvironmentalAlert:
    """Environmental alert"""
    timestamp: float
    level: AlertLevel
    parameter: str
    value: float
    threshold: float
    message: str
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'level': self.level.name,
            'parameter': self.parameter,
            'value': self.value,
            'threshold': self.threshold,
            'message': self.message,
            'acknowledged': self.acknowledged
        }


@dataclass
class EnvironmentalConfig:
    """Configuration for environmental monitoring"""
    
    # Monitoring intervals
    reading_interval_seconds: float = 10.0
    alert_check_interval_seconds: float = 5.0
    
    # Temperature thresholds (Celsius)
    temp_min_warning: float = 15.0
    temp_max_warning: float = 35.0
    temp_min_critical: float = 10.0
    temp_max_critical: float = 45.0
    temp_min_emergency: float = 5.0
    temp_max_emergency: float = 55.0
    
    # Humidity thresholds (%)
    humidity_min_warning: float = 30.0
    humidity_max_warning: float = 70.0
    humidity_min_critical: float = 20.0
    humidity_max_critical: float = 80.0
    
    # Pressure thresholds (kPa)
    pressure_min_warning: float = 95.0
    pressure_max_warning: float = 105.0
    pressure_min_critical: float = 90.0
    pressure_max_critical: float = 110.0
    
    # Vibration thresholds (g-force)
    vibration_warning: float = 0.1
    vibration_critical: float = 0.5
    vibration_emergency: float = 1.0
    
    # Noise thresholds (dB)
    noise_warning: float = 60.0
    noise_critical: float = 80.0
    
    # Air flow thresholds (m/s)
    airflow_min_warning: float = 0.1
    airflow_max_warning: float = 2.0
    airflow_min_critical: float = 0.05
    airflow_max_critical: float = 3.0
    
    # Data retention
    max_readings: int = 10000
    max_alerts: int = 1000
    
    # Sensor configuration
    enable_temperature: bool = True
    enable_humidity: bool = True
    enable_pressure: bool = True
    enable_vibration: bool = True
    enable_noise: bool = True
    enable_airflow: bool = True


class EnvironmentalMonitor:
    """Environmental monitoring system"""
    
    def __init__(self, config: EnvironmentalConfig):
        """Initialize environmental monitor"""
        self.config = config
        self.monitoring = False
        self.monitoring_thread = None
        
        # Data storage
        self.readings: List[EnvironmentalReading] = []
        self.alerts: List[EnvironmentalAlert] = []
        
        # Callbacks
        self.reading_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Current status
        self.current_reading: Optional[EnvironmentalReading] = None
        self.system_status = "STOPPED"
        
        logger.info("Environmental monitor initialized")
    
    def start_monitoring(self) -> bool:
        """Start environmental monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already running")
            return True
        
        try:
            self.monitoring = True
            self.system_status = "STARTING"
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.system_status = "RUNNING"
            logger.info("Environmental monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.monitoring = False
            self.system_status = "ERROR"
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop environmental monitoring"""
        if not self.monitoring:
            logger.warning("Monitoring not running")
            return True
        
        try:
            self.monitoring = False
            self.system_status = "STOPPING"
            
            # Wait for thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10.0)
            
            self.system_status = "STOPPED"
            logger.info("Environmental monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            self.system_status = "ERROR"
            return False
    
    def get_current_reading(self) -> Optional[EnvironmentalReading]:
        """Get current environmental reading"""
        return self.current_reading
    
    def get_recent_readings(self, count: int = 100) -> List[EnvironmentalReading]:
        """Get recent readings"""
        return self.readings[-count:] if len(self.readings) >= count else self.readings
    
    def get_active_alerts(self) -> List[EnvironmentalAlert]:
        """Get active (unacknowledged) alerts"""
        return [alert for alert in self.alerts if not alert.acknowledged]
    
    def get_all_alerts(self, count: int = 100) -> List[EnvironmentalAlert]:
        """Get all alerts"""
        return self.alerts[-count:] if len(self.alerts) >= count else self.alerts
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert"""
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_index}")
                return True
            else:
                logger.error(f"Invalid alert index: {alert_index}")
                return False
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    def add_reading_callback(self, callback: Callable):
        """Add callback for new readings"""
        self.reading_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for new alerts"""
        self.alert_callbacks.append(callback)
    
    def export_data(self, filename: str, include_alerts: bool = True) -> bool:
        """Export monitoring data to file"""
        try:
            data = {
                'config': self.config.__dict__,
                'readings': [reading.to_dict() for reading in self.readings],
                'system_status': self.system_status,
                'export_timestamp': time.time()
            }
            
            if include_alerts:
                data['alerts'] = [alert.to_dict() for alert in self.alerts]
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Data exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Monitoring loop started")
        
        while self.monitoring:
            try:
                # Take reading
                reading = self._take_reading()
                
                if reading:
                    # Store reading
                    self.readings.append(reading)
                    self.current_reading = reading
                    
                    # Limit stored readings
                    if len(self.readings) > self.config.max_readings:
                        self.readings = self.readings[-self.config.max_readings:]
                    
                    # Check for alerts
                    alerts = self._check_alerts(reading)
                    for alert in alerts:
                        self.alerts.append(alert)
                        self._notify_alert_callbacks(alert)
                    
                    # Limit stored alerts
                    if len(self.alerts) > self.config.max_alerts:
                        self.alerts = self.alerts[-self.config.max_alerts:]
                    
                    # Notify reading callbacks
                    self._notify_reading_callbacks(reading)
                
                # Wait for next reading
                time.sleep(self.config.reading_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)  # Brief pause before retry
        
        logger.info("Monitoring loop stopped")
    
    def _take_reading(self) -> Optional[EnvironmentalReading]:
        """Take environmental reading from sensors"""
        try:
            # Simulate sensor readings
            import random
            
            reading = EnvironmentalReading(
                timestamp=time.time(),
                temperature=25.0 + random.uniform(-5, 10) if self.config.enable_temperature else 25.0,
                humidity=50.0 + random.uniform(-20, 20) if self.config.enable_humidity else 50.0,
                pressure=101.3 + random.uniform(-5, 5) if self.config.enable_pressure else 101.3,
                vibration=random.uniform(0, 0.2) if self.config.enable_vibration else 0.0,
                noise_level=40.0 + random.uniform(0, 30) if self.config.enable_noise else 40.0,
                air_flow=0.5 + random.uniform(-0.3, 1.0) if self.config.enable_airflow else 0.5
            )
            
            return reading
            
        except Exception as e:
            logger.error(f"Failed to take reading: {e}")
            return None
    
    def _check_alerts(self, reading: EnvironmentalReading) -> List[EnvironmentalAlert]:
        """Check reading against thresholds and generate alerts"""
        alerts = []
        
        try:
            # Temperature alerts
            if self.config.enable_temperature:
                alerts.extend(self._check_temperature_alerts(reading))
            
            # Humidity alerts
            if self.config.enable_humidity:
                alerts.extend(self._check_humidity_alerts(reading))
            
            # Pressure alerts
            if self.config.enable_pressure:
                alerts.extend(self._check_pressure_alerts(reading))
            
            # Vibration alerts
            if self.config.enable_vibration:
                alerts.extend(self._check_vibration_alerts(reading))
            
            # Noise alerts
            if self.config.enable_noise:
                alerts.extend(self._check_noise_alerts(reading))
            
            # Air flow alerts
            if self.config.enable_airflow:
                alerts.extend(self._check_airflow_alerts(reading))
            
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
        
        return alerts
    
    def _check_temperature_alerts(self, reading: EnvironmentalReading) -> List[EnvironmentalAlert]:
        """Check temperature alerts"""
        alerts = []
        temp = reading.temperature
        
        if temp <= self.config.temp_min_emergency or temp >= self.config.temp_max_emergency:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.EMERGENCY,
                parameter="temperature",
                value=temp,
                threshold=self.config.temp_min_emergency if temp <= self.config.temp_min_emergency else self.config.temp_max_emergency,
                message=f"EMERGENCY: Temperature {temp:.1f}°C outside safe range"
            ))
        elif temp <= self.config.temp_min_critical or temp >= self.config.temp_max_critical:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.CRITICAL,
                parameter="temperature",
                value=temp,
                threshold=self.config.temp_min_critical if temp <= self.config.temp_min_critical else self.config.temp_max_critical,
                message=f"CRITICAL: Temperature {temp:.1f}°C outside acceptable range"
            ))
        elif temp <= self.config.temp_min_warning or temp >= self.config.temp_max_warning:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.WARNING,
                parameter="temperature",
                value=temp,
                threshold=self.config.temp_min_warning if temp <= self.config.temp_min_warning else self.config.temp_max_warning,
                message=f"WARNING: Temperature {temp:.1f}°C outside optimal range"
            ))
        
        return alerts
    
    def _check_humidity_alerts(self, reading: EnvironmentalReading) -> List[EnvironmentalAlert]:
        """Check humidity alerts"""
        alerts = []
        humidity = reading.humidity
        
        if humidity <= self.config.humidity_min_critical or humidity >= self.config.humidity_max_critical:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.CRITICAL,
                parameter="humidity",
                value=humidity,
                threshold=self.config.humidity_min_critical if humidity <= self.config.humidity_min_critical else self.config.humidity_max_critical,
                message=f"CRITICAL: Humidity {humidity:.1f}% outside acceptable range"
            ))
        elif humidity <= self.config.humidity_min_warning or humidity >= self.config.humidity_max_warning:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.WARNING,
                parameter="humidity",
                value=humidity,
                threshold=self.config.humidity_min_warning if humidity <= self.config.humidity_min_warning else self.config.humidity_max_warning,
                message=f"WARNING: Humidity {humidity:.1f}% outside optimal range"
            ))
        
        return alerts
    
    def _check_pressure_alerts(self, reading: EnvironmentalReading) -> List[EnvironmentalAlert]:
        """Check pressure alerts"""
        alerts = []
        pressure = reading.pressure
        
        if pressure <= self.config.pressure_min_critical or pressure >= self.config.pressure_max_critical:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.CRITICAL,
                parameter="pressure",
                value=pressure,
                threshold=self.config.pressure_min_critical if pressure <= self.config.pressure_min_critical else self.config.pressure_max_critical,
                message=f"CRITICAL: Pressure {pressure:.1f}kPa outside acceptable range"
            ))
        elif pressure <= self.config.pressure_min_warning or pressure >= self.config.pressure_max_warning:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.WARNING,
                parameter="pressure",
                value=pressure,
                threshold=self.config.pressure_min_warning if pressure <= self.config.pressure_min_warning else self.config.pressure_max_warning,
                message=f"WARNING: Pressure {pressure:.1f}kPa outside optimal range"
            ))
        
        return alerts
    
    def _check_vibration_alerts(self, reading: EnvironmentalReading) -> List[EnvironmentalAlert]:
        """Check vibration alerts"""
        alerts = []
        vibration = reading.vibration
        
        if vibration >= self.config.vibration_emergency:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.EMERGENCY,
                parameter="vibration",
                value=vibration,
                threshold=self.config.vibration_emergency,
                message=f"EMERGENCY: Excessive vibration {vibration:.2f}g detected"
            ))
        elif vibration >= self.config.vibration_critical:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.CRITICAL,
                parameter="vibration",
                value=vibration,
                threshold=self.config.vibration_critical,
                message=f"CRITICAL: High vibration {vibration:.2f}g detected"
            ))
        elif vibration >= self.config.vibration_warning:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.WARNING,
                parameter="vibration",
                value=vibration,
                threshold=self.config.vibration_warning,
                message=f"WARNING: Elevated vibration {vibration:.2f}g detected"
            ))
        
        return alerts
    
    def _check_noise_alerts(self, reading: EnvironmentalReading) -> List[EnvironmentalAlert]:
        """Check noise alerts"""
        alerts = []
        noise = reading.noise_level
        
        if noise >= self.config.noise_critical:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.CRITICAL,
                parameter="noise_level",
                value=noise,
                threshold=self.config.noise_critical,
                message=f"CRITICAL: High noise level {noise:.1f}dB detected"
            ))
        elif noise >= self.config.noise_warning:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.WARNING,
                parameter="noise_level",
                value=noise,
                threshold=self.config.noise_warning,
                message=f"WARNING: Elevated noise level {noise:.1f}dB detected"
            ))
        
        return alerts
    
    def _check_airflow_alerts(self, reading: EnvironmentalReading) -> List[EnvironmentalAlert]:
        """Check air flow alerts"""
        alerts = []
        airflow = reading.air_flow
        
        if airflow <= self.config.airflow_min_critical or airflow >= self.config.airflow_max_critical:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.CRITICAL,
                parameter="air_flow",
                value=airflow,
                threshold=self.config.airflow_min_critical if airflow <= self.config.airflow_min_critical else self.config.airflow_max_critical,
                message=f"CRITICAL: Air flow {airflow:.2f}m/s outside acceptable range"
            ))
        elif airflow <= self.config.airflow_min_warning or airflow >= self.config.airflow_max_warning:
            alerts.append(EnvironmentalAlert(
                timestamp=reading.timestamp,
                level=AlertLevel.WARNING,
                parameter="air_flow",
                value=airflow,
                threshold=self.config.airflow_min_warning if airflow <= self.config.airflow_min_warning else self.config.airflow_max_warning,
                message=f"WARNING: Air flow {airflow:.2f}m/s outside optimal range"
            ))
        
        return alerts
    
    def _notify_reading_callbacks(self, reading: EnvironmentalReading):
        """Notify reading callbacks"""
        for callback in self.reading_callbacks:
            try:
                callback(reading)
            except Exception as e:
                logger.error(f"Reading callback failed: {e}")
    
    def _notify_alert_callbacks(self, alert: EnvironmentalAlert):
        """Notify alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


def create_environmental_monitor(
    reading_interval: float = 10.0,
    temp_range: tuple = (20.0, 30.0),
    humidity_range: tuple = (40.0, 60.0)
) -> EnvironmentalMonitor:
    """
    Factory function to create environmental monitor
    
    Args:
        reading_interval: Seconds between readings
        temp_range: (min, max) temperature range for warnings
        humidity_range: (min, max) humidity range for warnings
        
    Returns:
        EnvironmentalMonitor instance
    """
    config = EnvironmentalConfig(
        reading_interval_seconds=reading_interval,
        temp_min_warning=temp_range[0],
        temp_max_warning=temp_range[1],
        humidity_min_warning=humidity_range[0],
        humidity_max_warning=humidity_range[1]
    )
    
    return EnvironmentalMonitor(config)


# Export main classes
__all__ = [
    'EnvironmentalMonitor',
    'EnvironmentalConfig',
    'EnvironmentalReading',
    'EnvironmentalAlert',
    'AlertLevel',
    'create_environmental_monitor'
]
