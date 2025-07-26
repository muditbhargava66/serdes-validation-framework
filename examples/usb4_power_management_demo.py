#!/usr/bin/env python3
"""
USB4 Power Management Demo

This script demonstrates USB4 power state management, monitoring, and validation
capabilities including power consumption measurement, wake event testing, and
thermal management validation.

Features demonstrated:
- Power state transition monitoring
- Power consumption validation for each USB4 state
- Wake event generation and timing analysis
- Thermal management and throttling validation
- USB-PD power delivery testing
- Comprehensive power management reporting
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.serdes_validation_framework.protocols.usb4 import (
    USB4LinkState,
    USB4PowerConfig,
    USB4PowerManager,
    USB4SignalMode,
    USB4WakeSource,
)


def main():
    """Main demonstration function"""
    print("USB4 Power Management Demo")
    print("=" * 50)
    
    # Create power management configuration
    config = USB4PowerConfig(
        signal_mode=USB4SignalMode.GEN2X2,
        sample_rate=1e9,
        capture_length=1000,
        monitor_duration=5.0,
        power_measurement_interval=0.1,
        thermal_monitoring=True,
        wake_event_monitoring=True,
        power_delivery_testing=True,
        throttling_testing=True
    )
    
    # Initialize power manager
    power_manager = USB4PowerManager(config)
    if not power_manager.initialize():
        print("Failed to initialize USB4 power manager")
        return
        
    try:
        # Demo 1: Power consumption validation
        print("\n1. Power Consumption Validation")
        print("-" * 30)
        
        for state in USB4LinkState:
            result = power_manager.validate_power_consumption(state)
            status = "PASS" if result.result.name == "PASS" else "FAIL"
            print(f"  {state.name}: {result.measured_value:.3f}W "
                  f"(limit: {result.limit_value:.3f}W) - {status}")
            
        # Demo 2: Power state monitoring
        print("\n2. Power State Monitoring")
        print("-" * 30)
        print("Monitoring power states for 3 seconds...")
        
        results = power_manager.monitor_power_states(3.0)
        
        print(f"  Total transitions: {results.total_transitions}")
        print(f"  Successful transitions: {results.successful_transitions}")
        print(f"  Power measurements: {len(results.power_measurements)}")
        print(f"  Wake events: {len(results.wake_events)}")
        print(f"  Thermal events: {len(results.thermal_events)}")
        
        # Display average power by state
        print("\n  Average Power by State:")
        for state, power in results.average_power_by_state.items():
            print(f"    {state.name}: {power:.3f}W")
            
        # Demo 3: Wake event testing
        print("\n3. Wake Event Testing")
        print("-" * 30)
        
        wake_sources = [
            USB4WakeSource.REMOTE_WAKE,
            USB4WakeSource.LOCAL_WAKE,
            USB4WakeSource.TIMER_WAKE,
        ]
        
        for wake_source in wake_sources:
            print(f"  Testing {wake_source.name}...")
            wake_events = power_manager.test_wake_events(wake_source, count=3)
            
            successful_wakes = sum(1 for event in wake_events if event.success)
            avg_wake_time = np.mean([event.wake_time for event in wake_events])
            avg_signal_quality = np.mean([event.signal_integrity for event in wake_events])
            
            print(f"    Success rate: {successful_wakes}/{len(wake_events)}")
            print(f"    Average wake time: {avg_wake_time*1000:.1f}ms")
            print(f"    Average signal quality: {avg_signal_quality:.3f}")
            
        # Demo 4: Thermal management testing
        print("\n4. Thermal Management Testing")
        print("-" * 30)
        
        # Test temperature profile from normal to critical and back
        temperature_profile = [25.0, 50.0, 70.0, 80.0, 90.0, 85.0, 70.0, 50.0, 25.0]
        
        print("Testing thermal management with temperature profile...")
        thermal_events = power_manager.validate_thermal_management(temperature_profile)
        
        print(f"  Thermal events generated: {len(thermal_events)}")
        
        for event in thermal_events:
            print(f"    {event.temperature:.1f}°C -> {event.thermal_state.name}")
            if event.throttling_level > 0:
                print(f"      Throttling: {event.throttling_level*100:.1f}%")
                print(f"      Performance impact: {event.performance_impact*100:.1f}%")
                print(f"      Mitigation: {event.mitigation_action}")
                
        # Demo 5: USB-PD power delivery testing
        print("\n5. USB-PD Power Delivery Testing")
        print("-" * 30)
        
        power_levels = [15.0, 30.0, 60.0, 100.0]
        pd_results = power_manager.test_power_delivery(power_levels)
        
        for result in pd_results:
            status = "PASS" if result.result.name == "PASS" else "FAIL"
            power_level = result.test_name.split()[1]  # Extract power level from test name
            print(f"  {power_level}: {status}")
            if hasattr(result, 'measured_value') and result.measured_value:
                print(f"    Measured: {result.measured_value:.1f}W")
                
        # Demo 6: Transition time measurement
        print("\n6. Power State Transition Times")
        print("-" * 30)
        
        transitions = [
            (USB4LinkState.U0, USB4LinkState.U1),
            (USB4LinkState.U1, USB4LinkState.U2),
            (USB4LinkState.U2, USB4LinkState.U3),
            (USB4LinkState.U3, USB4LinkState.U0),
        ]
        
        transition_times = power_manager.measure_transition_times(transitions)
        
        for (from_state, to_state), measured_time in transition_times.items():
            print(f"  {from_state.name} -> {to_state.name}: {measured_time*1000:.1f}ms")
            
        # Demo 7: Generate visualization
        print("\n7. Generating Power Analysis Plots")
        print("-" * 30)
        
        generate_power_plots(results, power_manager)
        print("  Power analysis plots saved to 'usb4_power_analysis.png'")
        
    finally:
        # Clean up
        power_manager.cleanup()
        print("\nUSB4 power manager demo completed successfully!")


def generate_power_plots(results, power_manager):
    """Generate power analysis visualization plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('USB4 Power Management Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Power consumption by state
    if results.average_power_by_state:
        states = list(results.average_power_by_state.keys())
        powers = list(results.average_power_by_state.values())
        state_names = [state.name for state in states]
        
        bars = ax1.bar(state_names, powers, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Average Power Consumption by State')
        ax1.set_ylabel('Power (W)')
        ax1.set_xlabel('USB4 Link State')
        
        # Add value labels on bars
        for bar, power in zip(bars, powers, strict=False):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{power:.3f}W', ha='center', va='bottom')
    
    # Plot 2: Power measurements over time
    if results.power_measurements:
        timestamps = [(m.timestamp - results.power_measurements[0].timestamp) 
                     for m in results.power_measurements]
        powers = [m.power for m in results.power_measurements]
        
        ax2.plot(timestamps, powers, 'b-', linewidth=1.5, alpha=0.7)
        ax2.set_title('Power Consumption Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power (W)')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: State transition times
    if results.transition_times:
        transition_labels = []
        transition_values = []
        
        for (from_state, to_state), time_val in results.transition_times.items():
            transition_labels.append(f"{from_state.name}→{to_state.name}")
            transition_values.append(time_val * 1000)  # Convert to ms
            
        bars = ax3.bar(range(len(transition_labels)), transition_values, 
                      color='orange', alpha=0.7)
        ax3.set_title('Power State Transition Times')
        ax3.set_ylabel('Transition Time (ms)')
        ax3.set_xlabel('State Transition')
        ax3.set_xticks(range(len(transition_labels)))
        ax3.set_xticklabels(transition_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, transition_values, strict=False):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}ms', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Wake event success rates
    if results.wake_events:
        wake_sources = {}
        for event in results.wake_events:
            source = event.wake_source.name
            if source not in wake_sources:
                wake_sources[source] = {'total': 0, 'successful': 0}
            wake_sources[source]['total'] += 1
            if event.success:
                wake_sources[source]['successful'] += 1
                
        if wake_sources:
            sources = list(wake_sources.keys())
            success_rates = [wake_sources[source]['successful'] / wake_sources[source]['total'] * 100
                           for source in sources]
            
            bars = ax4.bar(sources, success_rates, color='green', alpha=0.7)
            ax4.set_title('Wake Event Success Rates')
            ax4.set_ylabel('Success Rate (%)')
            ax4.set_xlabel('Wake Source')
            ax4.set_ylim(0, 100)
            ax4.set_xticklabels(sources, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates, strict=False):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('usb4_power_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
